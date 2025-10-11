# from FlowFouier import CopyTaskModel
from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_model
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
import random
import json

class Classifier(nn.Module):
    def __init__(self, src_vocab_size, src_padding_idx, d_model, n_head, dim_ff, n_layer, max_length, num_classes):
        super(Classifier, self).__init__()
        self.bert_config = BertConfig(
            vocab_size=src_vocab_size, pad_token_id=src_padding_idx,
            hidden_size=d_model, num_hidden_layers=n_layer,
            num_attention_heads=n_head, intermediate_size=dim_ff,
            max_position_embeddings=max_length,
        )
        # self.base_model = BertForMaskedLM(config=self.bert_config)
        self.base_model = BertModel(config=self.bert_config, add_pooling_layer=False)
        self.classifier = nn.Sequential(
            # nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(d_model,num_classes)
        )

    def weight_init(self, premodel_path, checkpoint=None):
        tmp = BertForMaskedLM(config=self.bert_config)
        if checkpoint:
            # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,f"checkpoint-{checkpoint}","pytorch_model.bin")))
            load_model(tmp, os.path.join(premodel_path, f"checkpoint-{checkpoint}", 'model.safetensors'))
        else:
            # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,"pytorch_model.bin")))
            load_model(tmp, os.path.join(premodel_path, 'model.safetensors'))
        self.base_model = tmp.bert
        del tmp
        # if frozen:
        #     self.base_model.eval()
        #     self.base_model.requires_grad_(requires_grad=False)

    def forward(self, src):
        memory = self.base_model(src).last_hidden_state[:,0,:]
        y_logit = self.classifier(memory)
        return {'y_logit':y_logit}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(inputs['src'])
        loss_cls = F.cross_entropy(output['y_logit'], inputs['y_label'])
        loss = loss_cls
        return (loss, {'y_logit':output['y_logit']}) if return_outputs else loss

def compute_metrics(pred):
    y_label = pred.label_ids
    y_pred = pred.predictions
    acc_y = (y_label==y_pred).mean()
    return {'accuracy_y':acc_y}
def preprocess_logits_for_metrics(logits, labels):
    y_logit = logits
    y_pred = y_logit.argmax(dim=-1)
    return y_pred  #, labels

gpu_count = torch.cuda.device_count()
print(f"There are {gpu_count} GPUs is available.")

def func(ps, max_len=256):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_len:
            break
    return ' '.join(r[0:max_len])

premodel_name = "BERT-ps"
# checkpoint = 44850
checkpoint = None

tokenizer_path = 'ps_tokenizer'

import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-T', type=str, default=None)
    parser.add_argument('--dataset', '-D', type=str, default='../dataset/public/DataCon2020ETA')
    parser.add_argument('--premodel', '-P', type=str, default='202406132201')

    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--eval_per_epoch', '-ev', type=int, default=1)
    parser.add_argument('--n_epochs', '-ep', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5)

    args = parser.parse_args()

    dataset_path = args.dataset
    pre_timestamp = args.premodel
    batch_size = args.batch_size
    eval_per_epoch = args.eval_per_epoch
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate

    timestamp = args.timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")

    premodel_path = os.path.join("model", f"{premodel_name}_{pre_timestamp}")
    print("Pretrained Model Path:", premodel_path)
    with open(os.path.join(premodel_path, 'hyperparameters.json')) as f:
        hyperparameters = json.load(f)
    print(hyperparameters)

    # load pretrain model
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    df_train = pd.read_csv(f'{dataset_path}/train.csv.gz', index_col=0)
    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', index_col=0)

    label2idx = dict(zip(df_train['label'].value_counts().index,range(len(df_train['label'].value_counts().index))))
    num_classes = len(label2idx)
    df_train['label_idx'] = df_train['label'].map(label2idx)
    df_test['label_idx'] = df_test['label'].map(label2idx)

    train_texts, train_labels = df_train['ps'].apply(func).tolist(), df_train['label_idx'].tolist()
    train_texts_ids = torch.tensor(tokenizer(
        train_texts, truncation=True, padding="max_length", max_length=hyperparameters['max_length'], return_special_tokens_mask=True
    )['input_ids'])
    trainset = Dataset.from_dict({'src': train_texts_ids, 'y_label': train_labels})
    trainset.set_format(type='torch', columns=['src', 'y_label'])
    trainset = trainset.shuffle()

    test_texts, test_labels = df_test['ps'].apply(func).tolist(), df_test['label_idx'].tolist()
    test_texts_ids = torch.tensor(tokenizer(
        test_texts, truncation=True, padding="max_length", max_length=hyperparameters['max_length'], return_special_tokens_mask=True
    )['input_ids'])
    testset = Dataset.from_dict({'src': test_texts_ids, 'y_label': test_labels})
    testset.set_format(type='torch', columns=['src', 'y_label'])
    testset = testset.shuffle()


    model_root = 'model-classifier'
    if not os.path.exists(os.path.join(model_root, timestamp)):
        os.mkdir(os.path.join(model_root, timestamp))
    if not os.path.exists(os.path.join(model_root, timestamp, 'BERT')):
        os.mkdir(os.path.join(model_root, timestamp, 'BERT'))

    info = {
        'pre_timestamp': pre_timestamp,
        'dataset': dataset_path,
        'num_classes': num_classes,
        'label2idx': label2idx
    }
    with open(os.path.join(model_root, timestamp, 'BERT', 'info.json'), 'w') as f:
        json.dump(info, f)


    bert_config = BertConfig(
        vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
        hidden_size=hyperparameters['d_model'], num_hidden_layers=hyperparameters['n_layer'],
        num_attention_heads=hyperparameters['n_head'], intermediate_size=hyperparameters['dim_ff'],
        max_position_embeddings=hyperparameters['max_length'],
    )
    tmp = BertForMaskedLM(config=bert_config).to(device)
    if checkpoint:
        # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,f"checkpoint-{checkpoint}","pytorch_model.bin")))
        load_model(tmp, os.path.join(premodel_path, f"checkpoint-{checkpoint}", 'model.safetensors'))
    else:
        # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,"pytorch_model.bin")))
        load_model(tmp, os.path.join(premodel_path, 'model.safetensors'))
    premodel = tmp.bert
    del tmp
    premodel.eval()

    X_emb_train = np.empty(shape=(0, hyperparameters['d_model']))
    for i in tqdm(range(0, len(train_texts_ids), batch_size)):
        X_emb_train = np.concatenate([
            X_emb_train,
            premodel(train_texts_ids[i:i + batch_size].to(device)).last_hidden_state[:, 0, :].cpu().detach().numpy()
        ])
    X_emb_test = np.empty(shape=(0, hyperparameters['d_model']))
    for i in tqdm(range(0, len(test_texts_ids), batch_size)):
        X_emb_test = np.concatenate([
            X_emb_test,
            premodel(test_texts_ids[i:i + batch_size].to(device)).last_hidden_state[:, 0, :].cpu().detach().numpy()
        ])

    training_results = []
    for model_type in ['logist', 'rf']:
        model_dir = os.path.join(model_root, timestamp, 'BERT', model_type)
        print(model_dir)
        if model_type == 'rf':
            model = RandomForestClassifier()
        elif model_type == 'logist':
            model = LogisticRegression()
        else:
            raise Exception("Unknown model_type.")
        model.fit(X=X_emb_train, y=train_labels)
        y_pred = model.predict(X=X_emb_test)
        print(accuracy_score(y_true=test_labels, y_pred=y_pred))
        print(confusion_matrix(y_true=test_labels, y_pred=y_pred))
        print(classification_report(y_true=test_labels, y_pred=y_pred, digits=4))
        training_result = {
            'model_type': model_type,
            'accuracy': accuracy_score(y_true=test_labels, y_pred=y_pred),
            'confusion_matrix': confusion_matrix(y_true=test_labels, y_pred=y_pred).tolist(),
            'classification_report': classification_report(y_true=test_labels, y_pred=y_pred, output_dict=True)
        }
        training_results.append(training_result)
        # save model
        joblib.dump(model, f"{model_dir}.joblib")

    for model_type in ['finetune', 'org']:
        model_dir = os.path.join(model_root, timestamp, 'BERT', model_type)
        print(model_dir)
        model = Classifier(
            src_vocab_size=len(tokenizer.get_vocab()), src_padding_idx=tokenizer.pad_token_id,
            d_model=hyperparameters['d_model'], n_head=hyperparameters['n_head'],
            dim_ff=hyperparameters['dim_ff'], n_layer=hyperparameters['n_layer'],
            max_length=hyperparameters['max_length'],
            num_classes=num_classes
        )#.to(device)
        if model_type == 'org':
            pass
        elif model_type == 'finetune':
            model.weight_init(premodel_path=premodel_path, checkpoint=checkpoint)
            model.base_model.requires_grad_(requires_grad=False)
            training_args = TrainingArguments(
                output_dir=model_dir,
                learning_rate=learning_rate, per_device_train_batch_size=batch_size, num_train_epochs=n_epochs*5,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy='epoch',
                save_strategy='no',
                label_names=['y_label'],
            )
            trainer = MyTrainer(
                model=model, args=training_args,
                train_dataset=trainset, eval_dataset=testset,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=compute_metrics
            )
            trainer.train()
            model.base_model.requires_grad_(requires_grad=True)
        else:
            raise Exception('Error model_type')

        training_args = TrainingArguments(
            output_dir=model_dir,
            learning_rate=learning_rate, per_device_train_batch_size=batch_size, num_train_epochs=n_epochs,
            logging_dir=os.path.join(premodel_path, 'logs'),
            logging_strategy='steps', logging_steps=int(trainset.num_rows / (batch_size  * gpu_count* eval_per_epoch)),
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='steps', eval_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
            save_strategy='no',
            label_names=['y_label'],
        )

        trainer = MyTrainer(
            model=model, args=training_args,
            train_dataset=trainset, eval_dataset=testset,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model()

        y_pred = []
        for i in tqdm(range(0, len(test_texts_ids), batch_size)):
            y_pred_tmp = model(test_texts_ids[i:i+batch_size].to(device))['y_logit'].argmax(1).cpu().numpy()
            y_pred = np.concatenate([y_pred, y_pred_tmp])
        print(accuracy_score(y_true=test_labels, y_pred=y_pred))
        print(confusion_matrix(y_true=test_labels, y_pred=y_pred))
        print(classification_report(y_true=test_labels, y_pred=y_pred, digits=4))

        training_result = {
            'model_type': model_type,
            'accuracy': accuracy_score(y_true=test_labels, y_pred=y_pred),
            'confusion_matrix': confusion_matrix(y_true=test_labels, y_pred=y_pred).tolist(),
            'classification_report': classification_report(y_true=test_labels, y_pred=y_pred, output_dict=True)
        }
        training_results.append(training_result)

    print("Training End.")
    for r in training_results:
        print(r['model_type'], r['accuracy'], r['confusion_matrix'])
    with open(os.path.join(model_root, timestamp, 'training_results_BERT.json'), 'w') as f:
        json.dump(training_results, f)