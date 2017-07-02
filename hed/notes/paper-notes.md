Crux of the idea from : https://arxiv.org/abs/1504.06375

0. Not all edge are created equal : Edges are tied to image scale, Edges at lower scales contribute to discovery of edges at higher scales - fine to coarse
1. Layers at different depth in a deep network operate at larger receptive field size (scale)
2. Force each layer to predict scale specific edge maps
3. Use fully convolution network to take in arbitrary sizes image and produce feature maps at different conv layer
4. Feed feature map from each layer to deconv (upsampling to GT size) and compare to GT with weighted binary cross entropy loss after sigmoid
5. Weighted average of maps from each layer (weights learned through training) and compare with GT with weighted binary cross entropy
6. Fine tune the entire network with combined loss 3. + 4.
7. weights of binary cross entropy either local or global since 90% of pixels are non-edge
8. Change GT to be the consensus of atleast three annotators. Noisy/small scale edge have lower consensus
9. Rotate images to 16 different angles + horizontal flipping to augment data since training sets are small (Authors provided the data-set which has augmentation applied)
10. No use of Mean-shift to find thinner edge/post processing
