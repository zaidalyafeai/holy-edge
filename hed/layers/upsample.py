import tensorflow as tf
import numpy as np


def bilinear_upsample(x, shape):

    """
    Fixing filter aguments of the kernal to performs Bilinear upsampling since the paper same is used in the paper
    """

    inp_shape = x.shape.as_list()
    ch = inp_shape[3]
    assert ch is not None

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        See https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/solve.py#L10
        Fix the parameters of the conv kernel to perform bilinear interpolation
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, shape, shape, 1], 'SAME')
    edge = shape * (shape - 1)
    deconv = deconv[:, edge:-edge, edge:-edge, :]

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)
    return deconv


def deconv_upsample(x, shape):

    """
    Fixing filter aguments of the kernal to performs Bilinear upsampling since the paper same is used in the paper
    """

    inp_shape = x.shape.as_list()
    out_channels = inp_shape[-1]

    shape = int(shape)
    filter_shape = 2 * shape

    weight_var = tf.Variable(tf.random_normal([filter_shape, filter_shape, out_channels, out_channels]), name="deconv_filter")

    out_shape = tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape, [1, shape, shape, 1], 'SAME')

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape

    deconv.set_shape(inp_shape)

    return deconv
