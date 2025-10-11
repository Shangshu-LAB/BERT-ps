from torch import nn
from torch.nn import functional as F
import torch

from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv, GINConv
import dgl

from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_model
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
import random
import json


class FS_Net(nn.Module):
    def __init__(self, emb_num, padding_idx, class_num, emb_size=128, hidden_size=128, layer_n=2, dropout=0.3):
        super(FS_Net, self).__init__()
        self.padding_idx = padding_idx
        self.embbedding = nn.Embedding(num_embeddings=emb_num, embedding_dim=emb_size)
        self.encoder = nn.GRU(input_size=emb_size,hidden_size=hidden_size,num_layers=layer_n,dropout=dropout,bidirectional=True)
        self.decoder = nn.GRU(input_size=hidden_size*layer_n*2,hidden_size=hidden_size,num_layers=layer_n,dropout=dropout,bidirectional=True)
        self.reconstruction = nn.Sequential(
            nn.Linear(hidden_size * 2, emb_num),
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden_size*layer_n*2*4,hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size,class_num),
        )

    def forward(self, x):
        maxlen = x.shape[1]
        x_emb = self.embbedding(x)
        enc_output, h = self.encoder(x_emb.permute(1,0,2))
        z_e = nn.Flatten()(h.permute(1,0,2))

        dec_input = z_e.tile((maxlen,1,1))
        dec_output, s = self.decoder(dec_input)
        z_d = nn.Flatten()(s.permute(1,0,2))

        p = self.reconstruction(dec_output.permute(1,0,2))

        z = torch.concat([z_e,z_d,torch.mul(z_e,z_d),torch.abs(z_e-z_d)],dim=1)
        z_c = self.cls(z)
        return {'src_logits':p, 'y_logit':z_c}

    def predict_proba(self, x):
        maxlen = x.shape[1]
        x_emb = self.embbedding(x)
        enc_output, h = self.encoder(x_emb.permute(1, 0, 2))
        z_e = nn.Flatten()(h.permute(1, 0, 2))

        dec_input = z_e.tile((maxlen, 1, 1))
        dec_output, s = self.decoder(dec_input)
        z_d = nn.Flatten()(s.permute(1, 0, 2))

        z = torch.concat([z_e, z_d, torch.mul(z_e, z_d), torch.abs(z_e - z_d)], dim=1)
        y_logit = self.cls(z).detach().cpu().numpy()
        return y_logit

    def predict(self, src):
        return self.predict_proba(src).argmax(1)


def get_Bursts(Packets):
    Bursts = []
    burst = [0]
    for i in range(1, len(Packets)):
        if int(tokenizer.decode(Packets[i - 1])[1:-1]) * int(tokenizer.decode(Packets[i])[1:-1]) < 0:
        # if int(tokenizer.decode(Packets[1])[1:-1])
            Bursts.append(burst)
            burst = [i]
        else:
            burst.append(i)
    Bursts.append(burst)
    return Bursts
def get_TIG(P):
    V, E, B = list(range(len(P[1:-1]))), [], get_Bursts(P[1:-1])
    for i in range(len(B)):
        b = B[i]
        if len(b)>1:
            for j in range(len(b)-1):
                E.append((b[j], b[j+1]))
    for i in range(len(B)-1):
        if len(B[i])==1 and len(B[i+1])==1:
            E.append((B[i][0],B[i+1][0]))
        else:
            E.append((B[i][0], B[i+1][0]))
            E.append((B[i][-1], B[i+1][-1]))
    G = dgl.graph(E)
    G.ndata['length'] = torch.tensor(P[1:-1])
    # G.ndata['length'] = torch.tensor([int(tokenizer.decode(i)[1:-1]) for i in P[1:-1]])
    return G
    # return {'nodes':V, 'edges':E, 'length': P} # 可以直接返回Graph
class TIGDataset(DGLDataset):
    def __init__(self, TIGs, labels, filename=''):
        super(TIGDataset, self).__init__(filename)
        self.TIGs = TIGs
        self.labels = labels
        self.num_rows = len(labels)
    def __getitem__(self, item):
        TIG = self.TIGs[item]
        return TIG, self.labels[item]
    def __len__(self):
        return len(self.labels)


class DApp_MLP(nn.Module):
    def __init__(self, in_feats, out_feats=64, layer_nums = 3):
        super(DApp_MLP,self).__init__()
        self.linear_layers =nn.ModuleList()
        for each in range(layer_nums):
            in_features= in_feats if each == 0 else out_feats
            self.linear_layers.append(nn.Linear(in_features=in_features, out_features=out_feats))
        self.activate = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(p=0.025)
    def forward(self, x):
        x1 = x
        for mod in self.linear_layers :
            x1 = mod(x1)
            x1 = self.activate(x1)
        x2 = self.batchnorm(x1)
        x3 = self.dropout(x2)
        return x3

class DApp_classifier(nn.Module):
    def __init__(self, nb_classes, gin_layer_num=3, gin_hidden_units=64, iteration_nums = 3,
                 graph_pooling_type='sum', neighbor_pooling_type='sum', iteration_first=True,
                 embedding=True, num_embeddings=4000):
        #DApp: 3个GIN,顺序级联在一起
        super(DApp_classifier,self).__init__()
        self.nb_classes = nb_classes
        self.gin_layer_num = gin_layer_num
        self.gin_hidden_uints = gin_hidden_units
        self.iteration_nums = iteration_nums

        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type= neighbor_pooling_type

        self.interation_first = iteration_first
        self.embedding = embedding
        self.embedding_dim = gin_hidden_units      #embedding的设置为gin的隐藏神经元个数

        if embedding:
            self.embedding_layer = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.embedding_dim)
            mlp = DApp_MLP(self.embedding_dim, gin_hidden_units, layer_nums=self.gin_layer_num)
        else:
            mlp = DApp_MLP(1,out_feats=gin_hidden_units,layer_nums= self.gin_layer_num)
        self.gin_layers = GINConv(apply_func=mlp, aggregator_type= self.neighbor_pooling_type, learn_eps=True)
        self.linear = nn.Linear(in_features=iteration_nums * gin_hidden_units,out_features=nb_classes)

    def forward(self, g):
        node_feature = g.ndata['length']
        node_feature = self.embedding_layer(node_feature) if self.embedding else node_feature.unsqueeze(-1).float()
        graph_feature_history = []
        layer = self.gin_layers
        for i in range(self.iteration_nums):
            node_feature = layer(g, node_feature)
            g.ndata['iterated_feature'] = node_feature
            if self.graph_pooling_type == 'sum':
                graph_feature = dgl.sum_nodes(g,'iterated_feature')
            elif self.graph_pooling_type == 'mean':
                graph_feature = dgl.mean_nodes(g,'iterated_feature')
            graph_feature_history.append(graph_feature)
        graph_features = torch.cat(graph_feature_history,-1)
        power = self.linear(graph_features)
        return {'y_logit':power}

    def predict_proba(self, src):
        g = dgl.batch([get_TIG(tokenizer(s)['input_ids']) for s in src]).to(device)
        y_logit = self.forward(g)['y_logit'].detach().cpu().numpy()
        return y_logit
    def predict(self, src):
        return self.predict_proba(src).argmax(1)



max_length = 256
tokenizer_path = 'ps_tokenizer'


import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"There are {gpu_count} GPUs is available.")
else:
    gpu_count = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-T', type=str, default=None)
    parser.add_argument('--dataset', '-D', type=str, default='../dataset_analysis/DataCon2020ETA')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--eval_per_epoch', '-ev', type=int, default=1)
    parser.add_argument('--n_epochs', '-ep', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5)

    args = parser.parse_args()
    dataset_path = args.dataset
    batch_size = args.batch_size
    eval_per_epoch = args.eval_per_epoch
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate

    timestamp = args.timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    df_train = pd.read_csv(f'{dataset_path}/train.csv.gz', index_col=0)
    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', index_col=0)

    label2idx = dict(zip(df_train['label'].value_counts().index, range(len(df_train['label'].value_counts().index))))
    num_classes = len(label2idx)
    df_train['label_idx'] = df_train['label'].map(label2idx)
    df_test['label_idx'] = df_test['label'].map(label2idx)

    model_root = 'model-classifier'
    if not os.path.exists(os.path.join(model_root, timestamp)):
        os.mkdir(os.path.join(model_root, timestamp))

    info = {'dataset': dataset_path, 'num_classes': num_classes, 'label2idx': label2idx}
    with open(os.path.join(model_root, timestamp, 'info.json'), 'w') as f:
        json.dump(info, f)

    training_results = []
    for model_type in ['FS-Net']:
        def func(ps, max_len=max_length):
            r = []
            for burst in ps.split(','):
                p_len, p_count = burst.split(':')
                r += [f"p{p_len}t"] * int(p_count)
                if len(r) > max_len:
                    break
            return ' '.join(r[0:max_len])
        train_texts, train_labels = df_train['ps'].apply(func).tolist(), df_train['label_idx'].tolist()
        train_inputs = torch.tensor(tokenizer(
            train_texts, truncation=True, padding="max_length", max_length=max_length,
            return_special_tokens_mask=True
        )['input_ids'])
        trainset = Dataset.from_dict({'src': train_inputs, 'y_label': train_labels})
        trainset.set_format(type='torch', columns=['src', 'y_label'])
        trainset = trainset.shuffle()

        test_texts, test_labels = df_test['ps'].apply(func).tolist(), df_test['label_idx'].tolist()
        test_inputs = torch.tensor(tokenizer(
            test_texts, truncation=True, padding="max_length", max_length=max_length,
            return_special_tokens_mask=True
        )['input_ids'])
        testset = Dataset.from_dict({'src': test_inputs, 'y_label': test_labels})
        testset.set_format(type='torch', columns=['src', 'y_label'])
        testset = testset.shuffle()

        model_dir = os.path.join(model_root, timestamp, model_type)
        print(model_dir)
        if model_type == 'FS-Net':
            model = FS_Net(
                emb_num=len(tokenizer.get_vocab()), padding_idx=tokenizer.pad_token_id,
                class_num=num_classes
            )
            class MyTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    output = model(inputs['src'])
                    tgt_labels = inputs['src']
                    # token_indexs = tgt_labels != model-classifier.module.padding_idx
                    token_indexs = tgt_labels != model.padding_idx
                    loss_tgt = F.cross_entropy(output['src_logits'][token_indexs], tgt_labels[token_indexs])
                    loss_cls = F.cross_entropy(output['y_logit'], inputs['y_label'])
                    loss = loss_tgt + loss_cls
                    return (loss, {'src_logits': output['src_logits'], 'y_logit': output['y_logit']}) if return_outputs else loss
            def compute_metrics(pred):
                src_labels, y_label = pred.label_ids
                src_preds, y_pred = pred.predictions
                token_indexs = src_labels != tokenizer.pad_token_id
                acc_tgt = (src_labels == src_preds)[token_indexs].sum() / token_indexs.sum()
                acc_y = (y_label == y_pred).mean()
                return {'acc_y': acc_y, 'acc_tgt': acc_tgt}
            def preprocess_logits_for_metrics(logits, labels):
                src_logits, y_logit = logits
                src_preds = src_logits.argmax(dim=-1)
                y_pred = y_logit.argmax(dim=-1)
                return src_preds, y_pred  # , labels
            training_args = TrainingArguments(
                output_dir=model_dir,
                learning_rate=learning_rate, per_device_train_batch_size=batch_size, num_train_epochs=n_epochs,
                logging_dir=os.path.join(model_dir, 'logs'),
                logging_strategy='steps', logging_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
                per_device_eval_batch_size=batch_size,
                evaluation_strategy='steps', eval_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
                save_strategy='no',
                label_names=['src', 'y_label'],
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
        for i in tqdm(range(0, len(test_inputs), batch_size)):
            y_pred_tmp = model(test_inputs[i:i + batch_size].to(device))['y_logit'].argmax(1).cpu().numpy()
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

    for model_type in ['GraphDApp']:
        def func(ps, max_len=max_length):
            r = []
            for burst in ps.split(','):
                p_len, p_count = burst.split(':')
                r += [f"p{p_len}t"] * int(p_count)
                if len(r) > max_len:
                    break
            return ' '.join(r)
        train_inputs, train_labels = df_train['ps'].apply(func).apply(lambda x: tokenizer(x)['input_ids']).apply(get_TIG).tolist(), df_train['label_idx'].tolist()
        trainset = TIGDataset(train_inputs, train_labels)
        test_inputs, test_labels = df_test['ps'].apply(func).apply(lambda x: tokenizer(x)['input_ids']).apply(get_TIG).tolist(), df_test['label_idx'].tolist()
        testset = TIGDataset(test_inputs, test_labels)

        model_dir = os.path.join(model_root, timestamp, model_type)
        print(model_dir)
        model = DApp_classifier(nb_classes=num_classes, num_embeddings=len(tokenizer.get_vocab()), embedding=True)
        class MyTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                output = model(inputs['src'])
                loss_cls = F.cross_entropy(output['y_logit'], inputs['y_label'])
                loss = loss_cls
                return (loss, {'y_logit': output['y_logit']}) if return_outputs else loss
        def compute_metrics(pred):
            y_label = pred.label_ids
            y_pred = pred.predictions
            acc_y = (y_label == y_pred).mean()
            return {'acc_y': acc_y}
        def preprocess_logits_for_metrics(logits, labels):
            y_logit = logits
            y_pred = y_logit.argmax(dim=-1)
            return y_pred  # , labels
        def data_collator(samples):
            graphs, labels = map(list, zip(*samples))
            return {'src': dgl.batch(graphs), 'y_label': torch.tensor(labels, dtype=torch.long)}
        training_args = TrainingArguments(
            output_dir=model_dir,
            learning_rate=learning_rate, per_device_train_batch_size=batch_size, num_train_epochs=n_epochs,
            logging_dir=os.path.join(model_dir, 'logs'),
            logging_strategy='steps', logging_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='steps', eval_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
            save_strategy='no',
            label_names=['y_label'],
        )
        trainer = MyTrainer(
            model=model, args=training_args,
            data_collator=data_collator,
            train_dataset=trainset, eval_dataset=testset,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model()

        y_pred = []
        for i in tqdm(range(0, len(test_inputs), batch_size)):
            y_pred_tmp = model(
                dgl.batch(test_inputs[i:i + batch_size]).to(device)
            )['y_logit'].argmax(1).cpu().numpy()
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
    with open(os.path.join(model_root, timestamp, 'training_results_DL.json'), 'w') as f:
        json.dump(training_results, f)
