# Adaptive Softmax for Keras

A Keras implementation of Adaptive Softmax and a variation of Differentiated Softmax as described in *[Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309)* by Grave, Joulin, Cissé, Grangier, and Jégou.

These variations of softmax exploit differences in word frequencies to provide substantial improvements in runtime performance, in particular with respect to training time.