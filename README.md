# DGCNN pytorch implementation

[[Paper]](https://arxiv.org/abs/1801.07829)   

## Overview
This is an unofficial & modified version of [DGCNN](https://github.com/WangYueFt/dgcnn) in pytorch.  

Currently the code is only about classification task for ModelNet40.

## Data Preparation

```bash
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
```

## Train
```python
python train.py
```
