from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import signatory
import itertools

import os
import random
import json




max_length = 256
tokenizer_path = 'ps_tokenizer'


import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"There are {gpu_count} GPUs is available.")
else:
    gpu_count = 1


def get_AppScanner(ps):
    upflow, downflow, binflow = [], [], []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        p_len, p_count = int(p_len), int(p_count)
        if p_len > 0:
            upflow += [p_len] * p_count
            binflow += [p_len] * p_count
        else:
            downflow += [-p_len] * p_count
            binflow += [-p_len] * p_count
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


def get_ETC_PS(ps):
    U0, D0, seq = [], [], []
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
    seq_CS = list(itertools.accumulate(seq))
    U0_CS = list(itertools.accumulate(U0))
    D0_CS = list(itertools.accumulate(D0))
    path = torch.tensor([seq, U0, D0, seq_CS, U0_CS, D0_CS]).permute((1, 0)).float()
    signature = signatory.signature(path.unsqueeze(0), 2).squeeze(0)
    return signature.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-T', type=str, default=None)
    parser.add_argument('--dataset', '-D', type=str, default='../dataset/public/DataCon2020ETA')
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


    for model_type in ['AppScanner', 'ETC-PS', 'FlowLens']:
        model_dir = os.path.join(model_root, timestamp, model_type)
        print(model_dir)

        if model_type == 'AppScanner':
            X_train, y_train = np.array(df_train['ps'].apply(get_AppScanner).tolist()), df_train['label'].map(label2idx)
            X_test, y_test = np.array(df_test['ps'].apply(get_AppScanner).tolist()), df_test['label'].map(label2idx)

        elif model_type == 'ETC-PS':
            X_train, y_train = np.array(df_train['ps'].apply(get_ETC_PS).tolist()), df_train['label'].map(label2idx)
            X_test, y_test = np.array(df_test['ps'].apply(get_ETC_PS).tolist()), df_test['label'].map(label2idx)

        elif model_type == 'FlowLens':
            QL = 4
            binPL_dist_count = Counter()
            for ps in df_train['ps']:
                binflow = []
                for burst in ps.split(','):
                    p_len, p_count = burst.split(':')
                    p_len, p_count = int(p_len), int(p_count)
                    if p_len > 0:
                        binflow += [p_len >> QL] * p_count
                    else:
                        binflow += [-((-p_len) >> QL)] * p_count
                binPL_dist_count += Counter(binflow)
            TruncationTab = sorted([i[0] for i in sorted(binPL_dist_count.items(), key=lambda x: x[1], reverse=True)[0:128]])
            def get_Flowlens(ps):
                binflow = []
                for burst in ps.split(','):
                    p_len, p_count = burst.split(':')
                    p_len, p_count = int(p_len), int(p_count)
                    if p_len > 0:
                        binflow += [p_len >> QL] * p_count
                    else:
                        binflow += [-((-p_len) >> QL)] * p_count
                binPL_dist_count = Counter(binflow)
                return [binPL_dist_count[binPL] for binPL in TruncationTab]
            X_train, y_train = np.array(df_train['ps'].apply(get_Flowlens).tolist()), df_train['label'].map(label2idx)
            X_test, y_test = np.array(df_test['ps'].apply(get_Flowlens).tolist()), df_test['label'].map(label2idx)
            FlowLens_info = {'TruncationTab':TruncationTab, 'QL':QL}
            with open(f"{model_dir}.info", "w") as f:
                json.dump(FlowLens_info, f)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, f"{model_dir}.joblib")


        y_pred = model.predict(X_test)
        print(accuracy_score(y_true=y_test, y_pred=y_pred))
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))
        print(classification_report(y_true=y_test, y_pred=y_pred, digits=4))
        training_result = {
            'model_type': model_type,
            'accuracy': accuracy_score(y_true=y_test, y_pred=y_pred),
            'confusion_matrix': confusion_matrix(y_true=y_test, y_pred=y_pred).tolist(),
            'classification_report': classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
        }
        training_results.append(training_result)


    print("Training End.")
    for r in training_results:
        print(r['model_type'], r['accuracy'], r['confusion_matrix'])
    with open(os.path.join(model_root, timestamp, 'training_results_ML.json'), 'w') as f:
        json.dump(training_results, f)
