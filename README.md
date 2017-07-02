## Holistically-Nested Edge Detection

Created by Harsimrat Sandhawalia

This repository contains tensorflow implementation of the [HED model](https://github.com/s9xie/hed). Details of hyperparamters are available in the [paper](https://arxiv.org/pdf/1504.06375.pdf)

    @InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
      Year  = {2015},
    }

## Get this repo
```
git clone https://github.com/harsimrat-eyeem/holy-edge.git
```

## Installing requirements
Its recommended to install the requirements in a [conda virtual invironment](https://conda.io/docs/using/envs.html#create-an-environment) 
```
cd holy-edge
pip install -r requirements.txt
```

## Setting up

Edit the [config file](https://github.com/harsimrat-eyeem/holy-edge/blob/master/hed/configs/hed.yaml) located at hed/configs/hed.yaml. Set the paths below. Make sure the directories exist and you have read/write permissions on them.
The HED model is trained on [augumented training](http://vcl.ucsd.edu/hed/HED-BSDS.tar) set created by the authors it is needed for traininig the model. 
```
# location where training data : http://vcl.ucsd.edu/hed/HED-BSDS.tar would be downloaded and decompressed
download_path: '<path>'
# location of snapshot and tensorbaord summary events 
save_dir: '<path>'
# location where to put the generated edgemaps during testing
test_output: '<path>'
```

## Training data & Models

0. Fetch VGG-16 models weights via git-lfs
```
git lfs fetch && git lfs pull
```
1. Download training data via following
```
python run-hed.py --download-data --config-file hed/configs/hed.yaml
```

## VGG-16 base model
VGG base model available [here](https://github.com/machrisaa/tensorflow-vgg) is used for producing multi-level features. The model is modified according with Section (3.) of the [paper](https://arxiv.org/pdf/1504.06375.pdf). Deconvolution layers are set with [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). The model uses single deconvolution layer in each side layers. Another implementation uses [stacked](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py#L35) bilinear deconvolution layers. In this implementation the upsampling parameters are learned while finetuning of the model. Pre-trained weights for [VGG-16](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) are hosted with [git-lfs](https://github.com/harsimrat-eyeem/holy-edge/blob/master/hed/models/vgg16.npy) in this repo. 

## Training
Launch training 
```
CUDA_VISIBLE_DEVICES=0 python run-hed.py --train --config-file hed/configs/hed.yaml
```
Launch tensorboard
```
tensorboard --logdir=<save_dir>
```

## Testing
Edit the snapshot you want to use for testing in hed/configs/hed.yaml
```
test_snapshot: <snapshot number>
```
```
CUDA_VISIBLE_DEVICES=1 python run-hed.py --test --config-file hed/configs/hed.yaml --gpu-limit 0.4
feh <test_output>
```
