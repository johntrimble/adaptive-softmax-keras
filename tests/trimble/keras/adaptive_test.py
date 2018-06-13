import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from trimble.keras import adaptive

def test_build_cluster_weight_shapes():
    assert [(1000, 5002), (256, 2000), (64, 3000)] == adaptive.build_cluster_weight_shapes([1000, 256, 64], [5000, 7000, 10000])
    assert [(8, 5002), (2, 2000), (2, 3000)] == adaptive.build_cluster_weight_shapes([8, 2, 2], [5000, 7000, 10000])

def test_build_cluster_projection_shapes():
    assert [None, (1000, 256), (1000, 64)] == adaptive.build_cluster_projection_shapes(1000, [1000, 256, 64])
    assert [(2048, 1000), (2048, 256), (2048, 64)] == adaptive.build_cluster_projection_shapes(2048, [1000, 256, 64])

def test_compute_child_cluster_masks():
    cutoffs = [3000, 5000, 7000, 10000]
    labels = np.array([[5586, 3971, 7741, 1349, 2822],
                       [3422, 1215, 6291, 7520, 1730],
                       [8577, 2887, 1507, 9086, 2399],
                       [4154, 7241, 1640, 3606, 9889],
                       [6227, 6129,  637, 8587, 1059],
                       [5079, 1630, 8016, 5110, 1078],
                       [2296, 1792, 7380, 1217, 3860],
                       [5159,  681, 8546, 2018, 5645],
                       [ 506, 3150, 6184, 6312, 2690],
                       [ 448,  982, 5918, 1128, 3960]], dtype='int32')

    with tf.Session() as session:
        inputs = tf.constant(labels)
        cluster_masks = session.run(adaptive.compute_child_cluster_masks(inputs, cutoffs))
        assert [3971, 3422, 4154, 3606, 3860, 3150, 3960] == labels[cluster_masks[0]].tolist()
        assert [5586, 6291, 6227, 6129, 5079, 5110, 5159, 5645, 6184, 6312, 5918] == labels[cluster_masks[1]].tolist()
        assert [7741, 7520, 8577, 9086, 7241, 9889, 8587, 8016, 7380, 8546] == labels[cluster_masks[2]].tolist()

def test_compute_logprob():
    # with timesteps
    with tf.Session() as sess:
        c1 = tf.constant(np.random.random((10, 5, 12)))
        c2 = tf.constant(np.random.random((10, 5, 10)))
        c3 = tf.constant(np.random.random((10, 5, 20)))
        result = sess.run(adaptive.compute_logprob([c1, c2, c3], [10, 20, 40]))
        # did we get the right shape?
        assert (10, 5, 40) == result.shape
        # do we have a valid probability distribution?
        prob_sum = np.sum(np.exp(result), axis=-1)
        assert np.all((prob_sum > 0.99999) & (prob_sum < 1.00001))

    # without timesteps
    with tf.Session() as sess:
        c1 = tf.constant(np.random.random((10, 12)))
        c2 = tf.constant(np.random.random((10, 10)))
        c3 = tf.constant(np.random.random((10, 20)))
        result = sess.run(adaptive.compute_logprob([c1, c2, c3], [10, 20, 40]))
        # did we get the right shape?
        assert (10, 40) == result.shape
        # do we have a valid probability distribution?
        prob_sum = np.sum(np.exp(result), axis=-1)
        assert np.all((prob_sum > 0.99999) & (prob_sum < 1.00001))

def test_AdaptiveSoftmaxProduceLogits_2d_inputs():
    vocab_size=10000
    cutoffs = [5000, 7000, 10000]

    data_input = Input(shape=(1000,), dtype='float32')
    labels_input = Input(shape=(1,), dtype='int32')
    adaptive_softmax = adaptive.AdaptiveSoftmaxProduceLogits(vocab_size, cutoffs=cutoffs)
    adaptive_softmax_out = adaptive_softmax([data_input, labels_input])

    # verify kernels for each cluster have correct dimensions
    assert (1000, 5002) == adaptive_softmax.cluster_kernels[0].shape
    assert (250, 2000) == adaptive_softmax.cluster_kernels[1].shape
    assert (62, 3000) == adaptive_softmax.cluster_kernels[2].shape

    # verify bais matrices have correct dimensions
    assert (5002,) == adaptive_softmax.cluster_biases[0].shape
    assert (2000,) == adaptive_softmax.cluster_biases[1].shape
    assert (3000,) == adaptive_softmax.cluster_biases[2].shape

    # verify projection matrices have correct dimensions
    assert adaptive_softmax.cluster_projections[0] is None
    assert (1000, 250) == adaptive_softmax.cluster_projections[1].shape
    assert (1000, 62) == adaptive_softmax.cluster_projections[2].shape

    retrieve_adaptive_softmax_output = K.function(
        [data_input, labels_input],
        adaptive_softmax_out)

    X = np.ones((10, 1000)).astype('float32')
    labels = np.array([[5842],
                       [2091],
                       [9793],
                       [8083],
                       [1473],
                       [3982],
                       [2364],
                       [8102],
                       [377],
                       [5615]]).astype('int32')
    outputs = retrieve_adaptive_softmax_output([X, labels])

    # verify the output shapes
    assert len(outputs) == len(cutoffs)
    assert (labels.shape[0], 5002) == outputs[0].shape
    assert (labels.shape[0], 2000) == outputs[1].shape
    assert (labels.shape[0], 3000) == outputs[2].shape

def test_AdaptiveSoftmaxProduceLogits_3d_inputs():
    vocab_size=10000
    cutoffs = [5000, 7000, 10000]

    data_input = Input(shape=(None,1000), dtype='float32')
    labels_input = Input(shape=(None,), dtype='int32')
    adaptive_softmax = adaptive.AdaptiveSoftmaxProduceLogits(vocab_size, cutoffs=cutoffs)
    adaptive_softmax_out = adaptive_softmax([data_input, labels_input])

    retrieve_adaptive_softmax_output = K.function(
        [data_input, labels_input],
        adaptive_softmax_out)

    X = np.ones((10, 5, 1000)).astype('float32')
    labels = np.array([[5586, 3971, 7741, 1349, 2822],
                       [3422, 1215, 6291, 7520, 1730],
                       [8577, 2887, 1507, 9086, 2399],
                       [4154, 7241, 1640, 3606, 9889],
                       [6227, 6129,  637, 8587, 1059],
                       [5079, 1630, 8016, 5110, 1078],
                       [2296, 1792, 7380, 1217, 3860],
                       [5159,  681, 8546, 2018, 5645],
                       [ 506, 3150, 6184, 6312, 2690],
                       [ 448,  982, 5918, 1128, 3960]], dtype='int32')
    outputs = retrieve_adaptive_softmax_output([X, labels])

    print([x.shape for x in outputs])

    # verify the output shapes
    assert len(outputs) == len(cutoffs)
    assert (labels.shape[0], 5, 5002) == outputs[0].shape
    assert (labels.shape[0], 5, 2000) == outputs[1].shape
    assert (labels.shape[0], 5, 3000) == outputs[2].shape

def test_AdaptiveSoftmaxProduceLogits_masking():
    vocab_size=10000
    cutoffs = [5000, 7000, 10000]

    data_input = Input(shape=(None,1000), dtype='float32')
    labels_input = Input(shape=(None,), dtype='int32')
    mask = tf.constant([[True,  True,  True,  True,  False],
                        [True,  True,  False, False, False],
                        [True,  True,  True,  True,  False],
                        [True,  True,  True,  False, False],
                        [True,  True,  True,  True,  False],
                        [True,  True,  True,  True,   True],
                        [True,  True,  True,  True,   True],
                        [True,  True,  True,  True,   True],
                        [True,  False, False, False, False],
                        [True,  True,  True,  True,  False]])
    mask = tf.Print(mask, [tf.shape(mask)], message="Mask mask shape: ")
    add_mask = Lambda(lambda x: x, mask=[mask, mask])
    inputs_masked = add_mask([data_input, labels_input])

    adaptive_softmax = adaptive.AdaptiveSoftmaxProduceLogits(vocab_size, cutoffs=cutoffs)
    adaptive_softmax_out = adaptive_softmax(inputs_masked)

    retrieve_adaptive_softmax_output = K.function(
        [data_input, labels_input],
        adaptive_softmax_out)

    X = np.ones((10, 5, 1000)).astype('float32')
    labels = np.array([[5586, 3971, 7741, 1349, 2822],
                       [3422, 1215, 6291, 7520, 1730],
                       [8577, 2887, 1507, 9086, 2399],
                       [4154, 7241, 1640, 3606, 9889],
                       [6227, 6129,  637, 8587, 1059],
                       [5079, 1630, 8016, 5110, 1078],
                       [2296, 1792, 7380, 1217, 3860],
                       [5159,  681, 8546, 2018, 5645],
                       [ 506, 3150, 6184, 6312, 2690],
                       [ 448,  982, 5918, 1128, 3960]], dtype='int32')
    outputs = retrieve_adaptive_softmax_output([X, labels])

    # verify the output shapes
    assert len(outputs) == len(cutoffs)
    assert (10, 5, 5002) == outputs[0].shape
    assert (10, 5, 2000) == outputs[1].shape
    assert (10, 5, 3000) == outputs[2].shape

def test_AdaptiveLogProb():
    vocab_size=10000
    cutoffs = [5000, 7000, 10000]

    data_input = Input(shape=(None,1000), dtype='float32')

    x = adaptive.AdaptiveSoftmaxProduceLogits(vocab_size, cutoffs=cutoffs)(data_input)
    x = adaptive.AdaptiveLogProb()(x)

    retrieve_adaptive_softmax_output = K.function([data_input], [x])

    outputs = retrieve_adaptive_softmax_output([np.random.random((2, 5, 1000)).astype('float32')])
    prob_sum = np.sum(np.exp(outputs), axis=-1)
    assert np.all((prob_sum > 0.99999) & (prob_sum < 1.00001))
