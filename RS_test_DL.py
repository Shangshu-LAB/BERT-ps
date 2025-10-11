from torch import nn
from torch.nn import functional as F
import torch

from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from safetensors.torch import load_model, load_file

from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv, GINConv
import dgl

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

    def predict_proba(self, src):
        x = torch.tensor(
            tokenizer(
                src, truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True
            )['input_ids']
        )  # .to(device)
        maxlen = x.shape[1]
        x_emb = self.embbedding(x.to(device))
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
            self.linear_layers.append(nn.Linear(in_features= in_features,out_features=out_feats))
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



def func(ps, max_len=256):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_len:
            break
    return ' '.join(r[0:max_len])

max_length = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

tokenizer_path = 'ps_tokenizer'

from datetime import datetime
import argparse

sample_n = 1000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='../dataset/public/DataCon2020ETA')
    parser.add_argument('--model', '-m', type=str, default='202408191454')
    parser.add_argument('--model_type', '-mt', type=str, default='GraphDApp')
    # parser.add_argument('--model_type', '-mt', type=str, default='FS-Net')
    parser.add_argument('--shuffle_type', '-st', type=str, default='loss')
    # parser.add_argument('--shuffle_type', '-st', type=str, default='retrans')
    # parser.add_argument('--shuffle_type', '-st', type=str, default='disorder')
    parser.add_argument('--rate', '-r', type=float, default=0.1)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
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

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    with open(os.path.join("model-classifier", timestamp, 'info.json')) as f:
        info = json.load(f)
    num_classes = info["num_classes"]
    label2idx = info["label2idx"]

    # dataset
    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', index_col=0)
    df_test['label_idx'] = df_test['label'].map(label2idx)

    if model_type in ['FS-Net']:
        model_path = os.path.join("model-classifier", timestamp, model_type, 'model.safetensors')
        test_texts, test_labels = df_test['ps'].apply(func).tolist(), df_test['label_idx'].tolist()

        model = FS_Net(
            emb_num=len(tokenizer.get_vocab()), padding_idx=tokenizer.pad_token_id,
            class_num=num_classes
        ).to(device)
        # load_model(model, model_path)
        model.load_state_dict(load_file(model_path))
        model.eval()

    elif model_type in ['GraphDApp']:
        model_path = os.path.join("model-classifier", timestamp, model_type, 'model.safetensors')
        test_texts, test_labels = df_test['ps'].apply(func).tolist(), df_test['label_idx'].tolist()
        # test_texts, test_labels = df_test['ps'].apply(func).apply(lambda x: tokenizer(x)['input_ids']).apply(get_TIG).tolist(), df_test['label_idx'].tolist()
        model = DApp_classifier(nb_classes=num_classes, num_embeddings=len(tokenizer.get_vocab()), embedding=True).to(device)
        load_model(model, model_path)
        model.eval()

    y_pred = np.array([], dtype=np.int_)
    print(len(test_texts))
    for i in range(0,len(test_texts),batch_size):
        y_pred_tmp = model.predict(test_texts[i:i+batch_size])
        y_pred = np.concatenate([y_pred,y_pred_tmp])

    df_test['label_idx'] = df_test['label'].astype(str).map(label2idx)
    y_true = df_test['label_idx'].to_numpy()
    # del pred #, test_texts, test_texts_ids
    print(accuracy_score(y_pred=y_pred,y_true=y_true))

    input_func = lambda batch: batch
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

