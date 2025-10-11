from torch import nn
from torch.nn import functional as F
import torch

from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from safetensors.torch import load_model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os
import random
import json
import joblib

from scipy.stats import norm, binomtest
import math
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from CertRobustness import Smooth

class Classifier_ML(nn.Module):
    def __init__(self, premodel, cls):
        super(Classifier_ML, self).__init__()
        self.premodel = premodel
        self.cls = cls
    def predict_proba(self, src):
        memory = self.premodel(src).last_hidden_state[:,0,:].detach().cpu()
        y_logit = self.cls.predict_proba(memory)
        return y_logit
    def predict(self, src):
        return self.predict_proba(src).argmax(1)
class Classifier_DL(nn.Module):
    def __init__(self, src_vocab_size, src_padding_idx, d_model, n_head, dim_ff, n_layer, max_length, num_classes):
        super(Classifier_DL, self).__init__()
        self.bert_config = BertConfig(
            vocab_size=src_vocab_size, pad_token_id=src_padding_idx,
            max_position_embeddings=max_length,
            hidden_size=d_model, num_hidden_layers=n_layer,
            num_attention_heads=n_head, intermediate_size=dim_ff,
        )
        self.base_model = BertModel(config=self.bert_config, add_pooling_layer=False)
        self.classifier = nn.Sequential(
            # nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(d_model,num_classes)
        )

    def weight_init(self, premodel_path, checkpoint=None, frozen=True):
        tmp = BertForMaskedLM(config=self.bert_config)
        if checkpoint:
            # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,f"checkpoint-{checkpoint}","pytorch_model.bin")))
            load_model(tmp, os.path.join(premodel_path, f"checkpoint-{checkpoint}", 'model.safetensors'))
        else:
            # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,"pytorch_model.bin")))
            load_model(tmp, os.path.join(premodel_path, 'model.safetensors'))
        self.base_model = tmp.bert
        del tmp
        if frozen:
            self.base_model.requires_grad_(requires_grad=False)
    def predict_proba(self, src):
        memory = self.base_model(src).last_hidden_state[:,0,:]
        y_logit = self.classifier(memory).detach().cpu().numpy()
        return y_logit
    def predict(self, src):
        return self.predict_proba(src).argmax(1)

    # def forward(self, src):
    #     memory = self.base_model(src).last_hidden_state[:,0,:]
    #     y_logit = self.classifier(memory)
    #     return {'y_logit':y_logit}



def func(ps, max_len=256):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_len:
            break
    return ' '.join(r[0:max_len])

max_length = 256
premodel_name = "BERT-ps"
tokenizer_path = 'ps_tokenizer'

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

from datetime import datetime
import argparse

sample_n = 1000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='../dataset/public/DataCon2020ETA')
    parser.add_argument('--model', '-m', type=str, default='202408191454')
    parser.add_argument('--model_type', '-mt', type=str, default='finetune')
    # parser.add_argument('--model_type', '-mt', type=str, default='rf')
    # parser.add_argument('--shuffle_type', '-st', type=str, default='loss')
    # parser.add_argument('--shuffle_type', '-st', type=str, default='retrans')
    parser.add_argument('--shuffle_type', '-st', type=str, default='disorder')
    parser.add_argument('--rate', '-r', type=float, default=0.1)
    parser.add_argument('--batch_size', '-b', type=int, default=200)
    parser.add_argument('--n0', '-n0', type=int, default=100)
    parser.add_argument('--n', '-n', type=int, default=5000)
    parser.add_argument('--alpha', '-a', type=int, default=0.01)
    parser.add_argument('--output', '-o', type=str, default='RS_test')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    dataset_path = args.dataset
    timestamp = args.model
    # org fintune frozen
    model_type = args.model_type
    shuffle_type = args.shuffle_type
    rate = args.rate
    batch_size = args.batch_size
    n0 = args.n0
    n = args.n
    alpha = args.alpha
    output_dir = args.output
    debug = args.debug
    # print(model_path, sigma, rate)

    # dataset
    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', index_col=0)

    with open(os.path.join("model-classifier", timestamp, 'BERT', 'info.json')) as f:
        info = json.load(f)
    pre_timestamp = info["pre_timestamp"]
    num_classes = info["num_classes"]
    label2idx = info["label2idx"]

    premodel_path = os.path.join("model", f"{premodel_name}_{pre_timestamp}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    with open(os.path.join(premodel_path, 'hyperparameters.json')) as f:
        hyperparameters = json.load(f)
    print(hyperparameters)

    if model_type in ['finetune', 'org']:
        model = Classifier_DL(
            src_vocab_size=len(tokenizer.get_vocab()), src_padding_idx=tokenizer.pad_token_id,
            d_model=hyperparameters['d_model'], n_head=hyperparameters['n_head'],
            dim_ff=hyperparameters['dim_ff'], n_layer=hyperparameters['n_layer'],
            max_length=hyperparameters['max_length'],
            num_classes = num_classes
        ).to(device)
        model_path = os.path.join("model-classifier", timestamp, 'BERT', model_type, 'model.safetensors')
        load_model(model, model_path)
        model.eval()
    elif model_type in ['rf', 'logist']:
        bert_config = BertConfig(
            vocab_size=len(tokenizer.get_vocab()), max_position_embeddings=hyperparameters['max_length'],
            hidden_size=hyperparameters['d_model'], num_hidden_layers=hyperparameters['n_layer'],
            num_attention_heads=hyperparameters['n_head'], intermediate_size=hyperparameters['dim_ff'],
        )
        tmp = BertForMaskedLM(config=bert_config).to(device)
        load_model(tmp, os.path.join(premodel_path, 'model.safetensors'))
        premodel = tmp.bert
        del tmp
        premodel.eval()
        model_path = os.path.join("model-classifier", timestamp, 'BERT', f"{model_type}.joblib")
        cls = joblib.load(model_path)
        model = Classifier_ML(premodel=premodel, cls=cls)

    input_func = lambda batch: torch.tensor(tokenizer(
        batch, truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True
    )['input_ids']).to(device)

    test_texts = df_test['ps'].apply(func).tolist()
    # test_texts_ids = torch.tensor(tokenizer(
    #     test_texts, truncation=True, padding="max_length", max_length=hyperparameters['max_length'],
    #     return_special_tokens_mask=True
    # )['input_ids'])

    y_pred = np.array([], dtype=np.int_)
    print(len(test_texts))
    for i in range(0,len(test_texts),batch_size):
        y_pred_tmp = model.predict(input_func(test_texts[i:i+batch_size]))
        y_pred = np.concatenate([y_pred,y_pred_tmp])

    df_test['label_idx'] = df_test['label'].astype(str).map(label2idx)
    y_true = df_test['label_idx'].to_numpy()
    # del pred #, test_texts, test_texts_ids
    print(accuracy_score(y_pred=y_pred,y_true=y_true))

    smoothed_model = Smooth(base_classifier=model, num_classes=num_classes, rate=rate, input_func=input_func)

    index_list = random.sample(range(len(df_test)), k=sample_n)

    result = []
    # start = datetime.now()
    for idx in tqdm(index_list):
        sample = func(df_test['ps'].iloc[idx])
        start = datetime.now()
        prediction, p = smoothed_model.certify(sample, n0=n0, n=n, alpha=alpha, batch_size=batch_size, shuffle_type=shuffle_type)
        end = datetime.now()
        item = {'label':y_true[idx], 'pred':y_pred[idx], 'smooth_pred': prediction, 'p':p, 'affected': y_pred[idx]!=prediction}
        # print('\t', idx, item, (end-start).total_seconds())
        result.append(item)
    # end = datetime.now()
    # print((end-start).total_seconds())
    pd.DataFrame(result).to_csv(f"{output_dir}/{model_type}_{shuffle_type}_{rate}.csv.gz", compression='gzip')
    print(model_path, rate, 'OK')

