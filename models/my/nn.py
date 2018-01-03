from my.general import flatten, reconstruct, add_wd, exp_mask

import numpy as np
import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"



def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):#, name_w='', name_b=''
    # if args is None or (nest.is_sequence(args) and not args):
    #     raise ValueError("`args` must be specified")
    # if not nest.is_sequence(args):
    #     args = [args]

    flat_args = [flatten(arg, 1) for arg in args]#[210,20]

    # if input_keep_prob < 1.0:
    #     assert is_train is not None
    flat_args = [tf.nn.dropout(arg, input_keep_prob) for arg in flat_args]
    
    total_arg_size = 0#[60]
    shapes = [a.get_shape() for a in flat_args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value
    # print(total_arg_size)
    # exit()
    dtype = [a.dtype for a in flat_args][0]        

    # scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(_WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
        if len(flat_args) == 1:
            res = tf.matmul(flat_args[0], weights)
        else: 
            res = tf.matmul(tf.concat(flat_args, 1), weights)
        if not bias:
            flat_out = res
        else:
            with tf.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                biases = tf.get_variable(
                    _BIAS_VARIABLE_NAME, [output_size],
                    dtype=dtype,
                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
            flat_out = tf.nn.bias_add(res, biases)    

    out = reconstruct(flat_out, args[0], 1)

    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    if wd:
        add_wd(wd)

    return out

def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)

        return out


def softsel(target, logits, mask=None, scope=None):
    """

    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask = mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out

def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob)
            prev = cur
        return cur

def conv1d(in_, filter_size, height, padding, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        in_ = tf.nn.dropout(in_, keep_prob)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, keep_prob=keep_prob, scope="conv1d_{}".format(height))
            outs.append(out)
        concat_out = tf.concat(outs, axis=2)
        return concat_out


if __name__ == '__main__':
    a = tf.Variable(np.random.random(size=(2,2,4)))
    b = tf.Variable(np.random.random(size=(2,3,4)))
    c = tf.tile(tf.expand_dims(a, 2), [1, 1, 3, 1])
    test = flatten(c,1)
    out = reconstruct(test, c, 1)
    d = tf.tile(tf.expand_dims(b, 1), [1, 2, 1, 1])
    e = linear([c,d,c*d],1,bias = False,scope = "test",)
    # f = softsel(d, e)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(test))
        print(sess.run(tf.shape(out)))
        exit()
        print(sess.run(tf.shape(a)))
        print(sess.run(a))
        print(sess.run(tf.shape(b)))
        print(sess.run(b))
        print(sess.run(tf.shape(c)))
        print(sess.run(c))  
        print(sess.run(tf.shape(d)))
        print(sess.run(d))
        print(sess.run(tf.shape(e)))
        print(sess.run(e))
