## Holistically-Nested Edge Detection

Created by Harsimrat Sandhawalia

This repository contains tensorflow implementation of the [HED model](https://github.com/s9xie/hed). Details of hyperparamters are available in the [paper](https://arxiv.org/pdf/1504.06375.pdf)

    @InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
      Year  = {2015},
    }

## VGG-16 base model
VGG base model available [here](https://github.com/machrisaa/tensorflow-vgg) is used for producing multi-level features. The model is modified according with Section 3. of the [paper](https://arxiv.org/pdf/1504.06375.pdf). Deconvolution layers are set with [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). The model uses single deconvolution layer in each side layers. Another implementation uses [stacked](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py#L35) bilinear deconvolution layers. In this implementation the upsampling parameters are learned while finetuning of the model. Pre-trained weights for [VGG-16](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) are hosted with [git-lfs]() in this repo. 
