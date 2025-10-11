from torch import nn
from torch.nn import functional as F
import torch

from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from safetensors.torch import load_model

from collections import Counter
import itertools
import signatory

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

def func(ps):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [p_len] * int(p_count)
    return ' '.join(r)

def get_AppScanner(ps, compressed=True):
    upflow, downflow, binflow = [], [], []
    if compressed:
        for burst in ps.split(','):
            p_len, p_count = burst.split(':')
            p_len, p_count = int(p_len), int(p_count)
            if p_len > 0:
                upflow += [p_len] * p_count
                binflow += [p_len] * p_count
            else:
                downflow += [-p_len] * p_count
                binflow += [-p_len] * p_count
    else:
        for p_len in ps.split(' '):
            if int(p_len) > 0:
                upflow.append(int(p_len))
                binflow.append(int(p_len))
            else:
                downflow.append(int(p_len))
                binflow.append(int(p_len))
    upflow, downflow, binflow = np.array(upflow), np.array(downflow), np.array(binflow)
    if len(upflow) > 0:
        item_upflow = [upflow.min(), upflow.max(), upflow.mean(), upflow.std()]
        item_upflow += np.percentile(upflow, [10, 20, 30, 40, 50, 60, 70, 80, 90]).tolist()
    else:
        item_upflow = [0] * 13
    if len(downflow) > 0:
        item_downflow = [downflow.min(), downflow.max(), downflow.mean(), downflow.std()]
        item_downflow += np.percentile(downflow, [10, 20, 30, 40, 50, 60, 70, 80, 90]).tolist()
    else:
        item_downflow = [0] * 13
    if len(binflow) > 0:
        item_binflow = [binflow.min(), binflow.max(), binflow.mean(), binflow.std()]
        item_binflow += np.percentile(binflow, [10, 20, 30, 40, 50, 60, 70, 80, 90]).tolist()
    else:
        item_binflow = [0] * 13
    return item_upflow + item_downflow + item_binflow


def get_ETC_PS(ps, compressed=True):
    U0, D0, seq = [], [], []
    if compressed:
        for burst in ps.split(','):
            p_len, p_count = burst.split(':')
            p_len, p_count = int(p_len), int(p_count)
            if p_len > 0:
                U0 += [p_len] * p_count
                D0 += [0] * p_count
                seq += [p_len] * p_count
            else:
                U0 += [0] * p_count
                D0 += [-p_len] * p_count
                seq += [-p_len] * p_count
    else:
        for p_len in ps.split(' '):
            if int(p_len) > 0:
                U0.append(int(p_len))
                D0.append(0)
                seq.append(int(p_len))
            else:
                U0.append(0)
                D0.append(-int(p_len))
                seq.append(-int(p_len))
    seq_CS = list(itertools.accumulate(seq))
    U0_CS = list(itertools.accumulate(U0))
    D0_CS = list(itertools.accumulate(D0))
    path = torch.tensor([seq, U0, D0, seq_CS, U0_CS, D0_CS]).permute((1, 0)).float()
    signature = signatory.signature(path.unsqueeze(0), 2).squeeze(0)
    return signature.tolist()


def get_Flowlens(ps, TruncationTab, QL, compressed=True):
    binflow = []
    if compressed:
        for burst in ps.split(','):
            p_len, p_count = burst.split(':')
            p_len, p_count = int(p_len), int(p_count)
            if p_len > 0:
                binflow += [p_len >> QL] * p_count
            else:
                binflow += [-((-p_len) >> QL)] * p_count
    else:
        for p_len in ps.split(' '):
            if int(p_len) > 0:
                binflow.append(int(p_len) >> QL)
            else:
                binflow.append(-((-int(p_len)) >> QL))
    binPL_dist_count = Counter(binflow)
    return [binPL_dist_count[binPL] for binPL in TruncationTab]


def func(ps, max_len=256):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [p_len] * int(p_count)
        if len(r) > max_len:
            break
    return ' '.join(r[0:max_len])

max_length = 256
premodel_name = "BERT"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

from datetime import datetime
import argparse

sample_n = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='../dataset/public/DataCon2020ETA')
    parser.add_argument('--model', '-m', type=str, default='202408191454')
    # parser.add_argument('--model_type', '-mt', type=str, default='AppScanner')
    # parser.add_argument('--model_type', '-mt', type=str, default='ETC-PS')
    parser.add_argument('--model_type', '-mt', type=str, default='FlowLens')
    # parser.add_argument('--shuffle_type', '-st', type=str, default='loss')
    # parser.add_argument('--shuffle_type', '-st', type=str, default='retrans')
    parser.add_argument('--shuffle_type', '-st', type=str, default='disorder')
    parser.add_argument('--rate', '-r', type=float, default=0.3)
    parser.add_argument('--batch_size', '-b', type=int, default=10000)
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
    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', compression='gzip', index_col=0)

    with open(os.path.join("model-classifier", timestamp, 'info.json')) as f:
        info = json.load(f)
    num_classes = info["num_classes"]
    label2idx = info["label2idx"]

    if model_type == 'AppScanner':
        get_feature = get_AppScanner
    elif model_type == 'ETC-PS':
        get_feature = get_ETC_PS
    elif model_type == 'FlowLens':
        with open(os.path.join("model-classifier", timestamp, f"{model_type}.info")) as f:
            FlowLens_info = json.load(f)
        TruncationTab, QL = FlowLens_info['TruncationTab'], FlowLens_info['QL']
        get_feature = lambda x, compressed=True: get_Flowlens(x, TruncationTab, QL, compressed)

    model_path = os.path.join("model-classifier", timestamp, f"{model_type}.joblib")
    model = joblib.load(model_path)


    # X_test, y_test = np.array(df_test['ps'].apply(get_AppScanner).tolist()), df_test['label'].map(label2idx)

    test_texts = np.array(df_test['ps'].apply(get_feature).tolist())

    y_pred = np.array([], dtype=np.int_)
    print(len(test_texts))
    for i in range(0,len(test_texts),batch_size):
        y_pred_tmp = model.predict(test_texts[i:i+batch_size])
        y_pred = np.concatenate([y_pred,y_pred_tmp])

    df_test['label_idx'] = df_test['label'].astype(str).map(label2idx)
    y_true = df_test['label_idx'].to_numpy()
    # del pred #, test_texts, test_texts_ids
    print(accuracy_score(y_pred=y_pred,y_true=y_true))

    # y_pred_tmp = self.base_classifier.predict(
    #     [get_feature(s, compressed=False) for s in samples_noised[i:i + batch_size]]
    # )
    input_func = lambda batch: [get_feature(s, compressed=False) for s in batch]
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

