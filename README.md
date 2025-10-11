# BERT-ps
The repository of paper "Robustness Matters: Pre-Training can Enhance the Performance of Encrypted Traffic Analysis", accepted by TIFS'25.

## requirements
Pytorch >= 2.3.0

Transformers >= 4.41.1

Zeek >= 6.0.3

## Pre-processing

Use the Zeek plug-in to parse the pcap file and generate Zeek logs. In detail, ps.log records the packet length sequence.

## Pre-training
Pre-training relies on large-scale unlabeled data of packet length sequences. 
This stage enables effective representation learning of the network traffic information from packet sequences.
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python BERT_pretrain.py -D pretrain_data_dir -d 768 -nh 12 -l 12 -f 3072 -b batch_size -ev 10 -sv 5 -ep n_epochs -lr 5e-5
```
The pre-trained model is saved in "model/{timestamp}" 

## Fine-tuning

We performs supervised fine-tuning (SFT) on a small set of labeled dataset regarding specific downstream tasks.

```python
python classifier_BERT_trainer.py -T timestamp -D dataset_dir -o output -b batch_size -ev eval_per_epoch -sv save_per_epoch -ep n_epochs -lr learning_rate 
```
The fine-tuned model is saved in "model-classifier/{timestamp}"

In addition, ML-based baselines are achieved in classifier_ML_trainer.py, including AppScanner, ETC-PS, and FlowLens.
```python
python classifier_ML_trainer.py -T timestamp -D dataset_dir -o output
```

Besides, DL-based baselines are achieved in classifier_DL_trainer.py, including FS-Net and GraphDApp.
```python
python classifier_DL_trainer.py -T timestamp -D dataset_dir -o output -b batch_size -ev eval_per_epoch -ep n_epochs -lr learning_rate 
```


## Robustness analysis
For BERT-ps, we performs robustness test on the test set.
For each sample in test set, we calculate its $$p_A$$ and $$p_B$$ using Monte Carlo method.
There are 3 type of network shuffle, including packet loss, retransmission, and disorder with different shuffle rate.
```python
python RS_test_BERT.py -D dataset -m model -mt model_type -st shuffle_type -r rate -n0 n0 -n n -b batch_size -o output -N n_sample
```
After this process, we can calculate the $$\Delta\hat{p} = \underline{p_A}-\overline{p_B}$$ for each sample and then generate the PA-curve and compute the PA-aera.

In addition, the robustness test of ML-based baselines and DL-based baselines are achieved by RS_test_ML.py and RS_test_DL.py, respectively.
```python
python RS_test_ML.py -D dataset -m model -mt model_type -st shuffle_type -r rate -n0 n0 -n n -b batch_size -o output -N n_sample
python RS_test_DL.py -D dataset -m model -mt model_type -st shuffle_type -r rate -n0 n0 -n n -b batch_size -o output -N n_sample
```
