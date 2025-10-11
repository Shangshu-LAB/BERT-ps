from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import torch
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
import itertools
import random



min_pkts = 10

import argparse


max_length = 256

def func(ps):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [p_len] * int(p_count)
        if len(r) > max_length+1:
            break
    return ' '.join(r[0:max_length+1])

model_name = "BERT"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='../dataset_analysis/school/log/conn')
    # parser.add_argument('--dataset', '-D', type=str, default='/media/roots/SanDisk/conn')
    parser.add_argument('--tokenizer', '-T', type=str, default='ps_tokenizer')

    parser.add_argument('--d_model', '-d', type=int, default=256)  # 32
    parser.add_argument('--n_head', '-nh', type=int, default=4)  # 4
    parser.add_argument('--n_layer', '-l', type=int, default=4)  # 4
    parser.add_argument('--dim_ff', '-f', type=int, default=1024)  # 64

    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--eval_per_epoch', '-ev', type=int, default=4)
    parser.add_argument('--save_per_epoch', '-sv', type=int, default=1)
    parser.add_argument('--n_epochs', '-ep', type=int, default=2)
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5)
    parser.add_argument('--test_size', '-ts', type=float, default=0.01)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    dataset_path = args.dataset
    tokenizer_path = args.tokenizer

    d_model = args.d_model
    n_head = args.n_head
    dim_ff = args.dim_ff
    n_layer = args.n_layer

    batch_size = args.batch_size
    eval_per_epoch = args.eval_per_epoch
    save_per_epoch = args.save_per_epoch
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    test_size = args.test_size
    debug = args.debug

    gpu_count = torch.cuda.device_count()
    print(f"There are {gpu_count} GPU is available.")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_path = os.path.join("model", f"{model_name}_{timestamp}")
    print(model_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if not os.path.exists(os.path.join(tokenizer_path, 'vocab.txt')) or not os.path.exists(os.path.join(tokenizer_path, 'config.json')):
        raise Exception(f"Error in tokenizer path (--tokenizer/-t).")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(model_path)
    def encode(examples):
        return tokenizer(examples['src'], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)


    hyperparameters = {
        'd_model': d_model, 'n_head': n_head, 'dim_ff': dim_ff, 'n_layer': n_layer,
        'max_length': max_length,
        'tokenizer': tokenizer_path, 'dataset': dataset_path,
    }

    filepaths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    if debug:
        df = pd.read_csv(filepaths[0], index_col=0, compression='gzip').iloc[0:100000]
        df = df[(df['orig_pkts']+df['resp_pkts']>=min_pkts)&(df['proto'].isin(['tcp','udp']))]
        df['text'] = df['ps'].apply(func, axis=1)
    else:
        df = pd.DataFrame()
        for filepath in filepaths[0:1]:
            print(f"Reading {filepath} ...", end=' ')
            start = datetime.now()
            df_tmp = pd.read_csv(filepath, index_col=0, compression='gzip')
            df_tmp = df_tmp[(df_tmp['orig_pkts']+df_tmp['resp_pkts']>=min_pkts)&(df_tmp['proto'].isin(['tcp', 'udp']))].iloc[0:1000000]
            print(f"Mapping ...", end=' ')
            df_tmp['text'] = df_tmp['ps'].apply(func)
            df = pd.concat([df, df_tmp])
            end = datetime.now()
            print(f"Done in {(end-start).total_seconds()}")
    print(f"There are {len(df)} rows in totals.")

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Trainset rows: {len(df_train)}, Testset rows: {len(df_test)}")

    trainset = Dataset.from_dict({'src': df_train['text'].to_list()})
    testset = Dataset.from_dict({'src': df_test['text'].to_list()})
    print("Tokenizing ...", end=' ')
    start = datetime.now()
    trainset = trainset.map(encode, batched=True)
    testset = testset.map(encode, batched=True)
    end = datetime.now()
    print(f"Done in {(end - start).total_seconds()}s.")

    trainset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'])
    testset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'])
    trainset = trainset.shuffle()
    testset = testset.shuffle()

    model_config = BertConfig(
        vocab_size=len(tokenizer.get_vocab()), max_position_embeddings=max_length,
        hidden_size=hyperparameters['d_model'], # 256/512/768/1024
        num_hidden_layers=hyperparameters['n_layer'], # 3/6/12/24
        num_attention_heads=hyperparameters['n_head'], # 4/8/12/16,
        intermediate_size=hyperparameters['dim_ff'], # 3072
    )
    model = BertForMaskedLM(config=model_config)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"There are {model_size/(1000*1000):.2f}M parameters in the model.")

    hyperparameters['model_size'] = model_size
    with open(os.path.join(model_path,'hyperparameters.json'), 'w') as func:
        json.dump(hyperparameters, func)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        learning_rate=learning_rate,

        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,

        logging_dir=os.path.join(model_path, 'logs'),
        # logging_strategy='epoch',
        logging_strategy='steps',
        logging_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
        # logging_steps=100,

        per_device_eval_batch_size=batch_size,
        evaluation_strategy='steps',
        eval_steps=int(trainset.num_rows / (batch_size * gpu_count * eval_per_epoch)),
        # eval_steps=100,

        # save_strategy='epoch',
        save_steps = int(trainset.num_rows / (batch_size * gpu_count * save_per_epoch)),

        # fsdp=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        data_collator=data_collator,
        train_dataset=trainset,
        eval_dataset=testset,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model()