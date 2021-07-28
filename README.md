# Testability-Aware Low Power Controller Design with Evolutionary Learning
This repo contains the source code of Testability-Aware Low Power Controller Design with Evolutionary Learning, ITC 2021.

The entry to the core algorithm is [ga.py](ga.py). Other files are not related to the paper and can be ignored.
# BNN-Testing
Created on 6/11/2020. The repo has been updated again on 4/09/2020.
Try BNN for lossless testing compression.
This repo only contains only a rough implementation of a binarized auto-encoder for compressing the test cubes.

The codes are referred from [jiecaoyu/XNOR-Net-PyTorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)

## Comparasion BNN with EDT
* From high-level pespective, they are the same, as BNN can be seem as a stacted XOR Net structure where its parameters should be learned from data.
* 1-layer decoder of BNN is exactly a XOR network.
  
## GA for EDT structure search
* Using GA to search an optimal XOR matrix for EDT, which are more effective than random XOR matrix.

### Consider the initialization of XORNet
How to initialize the XORNet is important. Usually, we need the matrix to be orthogonal. And we might refer to this [Xavier Initialization paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).




