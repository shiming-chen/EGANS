# EGANS

This is the codes of paper "**EGANS: Evolutionary Generative Adversarial Network Search for Zero-Shot Learning**" accepted to *IEEE Transactions on Evolutionary Computation (TEC)*. 

### Requirements
The code implementation of **EGANS** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.10.6.

We use CLSWGAN as the baseline to verify our method and conduct experiments on four popular ZSL benchmarks: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [FLO](https://tensorflow.google.cn/datasets/catalog/oxford_flowers102) and [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip). 

##  Evolution Generator Architecture Search

### Generator Architecture Searching Script

```
$ python clswgan_G_search.py
```

## Evolution Discriminator Architecture Search

### Discriminator Architecture Searching Script

```
$ python clswgan_D_search.py
```


## Zero-shot prediction
### Model retrain and final prediction

```
$ python clswgan_retrain.py
```

## Citation
If this work is helpful for you, please cite our paper.

```
@ARTICLE{10225587,
  author={Chen, Shiming and Chen, Shuhuang and Hou, Wenjin and Ding, Weiping and You, Xinge},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={EGANS: Evolutionary Generative Adversarial Network Search for Zero-Shot Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  keywords={Computer architecture;Generators;Generative adversarial networks;Visualization;Training;Semantics;Optimization;Evolutionary neural architecture search;zero-shot learning;generative adversarial networks},
  doi={10.1109/TEVC.2023.3307245}}
```
## Contact
If you have any questions, please drop email to gchenshiming@gmail.com or shuhuangchen@hust.edu.cn.
