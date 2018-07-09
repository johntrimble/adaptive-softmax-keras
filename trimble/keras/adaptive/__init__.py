import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras import initializers
from keras import regularizers

def build_default_capacities(max_capacity, cutoffs, min_capacity=2):
    return [max(min_capacity, max_capacity >> (2*i)) for i in range(len(cutoffs))]

def build_cluster_weight_shapes(capacities, cutoffs, hierarchical=True):
    number_child_clusters = len(cutoffs) - 1
    shapes = []
    for (start, end, capacity) in zip([0]+cutoffs, cutoffs, capacities):
        output_dim = end - start
        if start == 0 and hierarchical:
            # this is the head cluster, it contains a prefix of our words (ordered by decreasing frequency)
            # along with a node for each child cluster. This means it's output dimension will be the number
            # of words in the head plus the number of child clusters
            output_dim = output_dim + number_child_clusters
        shapes.append((capacity, output_dim))
    return shapes

def build_cluster_projection_shapes(input_size, capacities):
    shapes = []
    for capacity in capacities:
        if capacity == input_size:
            shapes.append(None)
        else:
            shapes.append((input_size, capacity))
    return shapes

def compute_child_cluster_masks(labels, cutoffs, mask=None):
    less_than_masks = [tf.less(labels, c) for c in cutoffs[1:-1]]
    greater_than_masks = [tf.greater_equal(labels, cutoffs[0])]
    greater_than_masks.extend([tf.logical_not(less_than) for less_than in less_than_masks])

    cluster_masks = [tf.logical_and(greater, less)
                     for greater, less in zip(greater_than_masks, less_than_masks)]
    cluster_masks.append(greater_than_masks[-1])

    if not mask is None:
        cluster_masks = [tf.logical_and(cluster_mask, mask) for cluster_mask in cluster_masks]

    return cluster_masks

def compute_cluster_labels(labels, child_cluster_masks, cutoffs, mask=None):
    cluster_labels = []
    head_labels = labels

    for i in range(len(child_cluster_masks)):
        child_cluster_idx = cutoffs[0] + i
        child_cluster_mask = child_cluster_masks[i]

        # labels for this cluster
        child_labels = tf.boolean_mask(labels, child_cluster_mask)
        child_labels = tf.subtract(child_labels, cutoffs[i])
        cluster_labels.append(child_labels)

        # update head labels indicating which child cluster the category resides in
        head_labels = tf.where(child_cluster_mask,
                               tf.fill(tf.shape(head_labels),
                                       tf.constant(child_cluster_idx, dtype='int32')),
                               head_labels)
    if not mask is None:
        head_labels = tf.boolean_mask(head_labels, mask)

    # Due to application of masks to our inputs, we will naturally lose the
    # steps dimension for the inputs of our tail clusters. We remove this
    # dimension from the head input as well for consistency.
    head_labels = K.reshape(head_labels, K.constant([-1], dtype='int32'))

    cluster_labels.insert(0, head_labels)
    return cluster_labels

def compute_cluster_inputs(inputs, child_cluster_masks, cutoffs, mask=None):
    # input for this cluster
    child_inputs = []
    for i in range(len(child_cluster_masks)):
        child_cluster_mask = child_cluster_masks[i]
        x = tf.boolean_mask(inputs, child_cluster_mask)
        child_inputs.append(x)

    if not mask is None:
        inputs = tf.boolean_mask(inputs, mask)

    # Due to application of masks to our inputs, we will naturally lose the
    # steps dimension for the inputs of our tail clusters. We remove this
    # dimension from the head input as well for consistency.
    if len(K.int_shape(inputs)) > 2:
        shape = K.shape(inputs)
        new_shape = K.concatenate([K.constant([-1], dtype='int32'), shape[2:]])
        inputs = K.reshape(inputs, new_shape)

    return [inputs] + child_inputs

def compute_logits(cluster_projections, cluster_kernels, cluster_biases, cluster_inputs):
    """
    # Arguments
        cluster_projections: List of k projection tensors, one for each cluster,
            with shape `(cluster_inputs[k].shape[-1], cluster_kernsl[k].shape[0])`.
            If `cluster_inputs[k].shape[-1] == cluster_kernsl[k].shape[0]` for
            a given cluster, then cluster_projections[k] may be None.
        cluster_kernels: List of k tensors, one for each cluster.
        cluster_biases: List of k tensors, one for each cluster, with shape
            `(cluster_kernels[k].shape[-1],)`
        cluster_inputs: List of k tensors, one for each cluster, with shape
            `(samples, ..., features)`. Typically, this is just the output from
            the previous layer repeated for each cluster.

    # Returns
        k tensors, one for each cluster, with shape
            `(samples, ..., cluster_kernels[k].shape[-1])`.
    """
    outputs = []
    for i in range(len(cluster_inputs)):
        projection = cluster_projections[i]
        kernel = cluster_kernels[i]
        bias = cluster_biases[i]
        cluster_input = cluster_inputs[i]
        x = cluster_input
        if projection:
            x = K.dot(cluster_input, projection)
        x = K.dot(x, kernel)
        if not bias is None:
            x = K.bias_add(x, bias)
        outputs.append(x)
    return outputs

def compute_adaptive_loss(cluster_projections, cluster_kernels, cluster_biases, a_prev, labels, cutoffs, mask=None):
    # get relevant logits for each cluster
    child_cluster_masks = compute_child_cluster_masks(labels, cutoffs, mask=mask)
    cluster_labels = compute_cluster_labels(labels, child_cluster_masks, cutoffs, mask=mask)
    cluster_inputs = compute_cluster_inputs(a_prev, child_cluster_masks, cutoffs, mask=mask)
    logits = compute_logits(cluster_projections,
                            cluster_kernels,
                            cluster_biases,
                            cluster_inputs)

    # compute loss

    # cluster_inputs and cluster_labels will no longer have a time dimension at
    # this point, meaning the head cluster will have batch_size*sequence_length
    # for it's first dimension. We will need this value later to average the
    # cost.
    batch_size_seq_length_product = tf.shape(cluster_inputs[0])[0]

    total_cost = None
    for logits, labels in zip(logits, cluster_labels):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cost = tf.reduce_sum(cost)
        if total_cost is None:
            total_cost = cost
        else:
            total_cost = tf.add(total_cost, cost)

    cost_mean = tf.divide(total_cost, tf.cast(batch_size_seq_length_product, 'float32'))
    return cost_mean

def compute_prob(clusters, cutoffs):
    # get prob for each cluster
    cluster_probs = [K.softmax(cluster_logits) for cluster_logits in clusters]

    # adjust prob for child clusters to account for probability
    # of picking child
    head_cluster_prob = cluster_probs[0]
    for i in range(len(cutoffs) - 1):
        child_idx = cutoffs[0]
        # the probability of choosing the ith child
        child_prob = head_cluster_prob[..., child_idx]
        child_prob = tf.expand_dims(child_prob, axis=-1)
        # multiply probability of choosing child to every
        # item in child cluster
        cluster_probs[i+1] = tf.multiply(cluster_probs[i+1], child_prob)

    # combine everything into flattened prob
    head_categories_prob = head_cluster_prob[..., 0:(cutoffs[0])]
    return tf.concat([head_categories_prob] + cluster_probs[1:], axis=-1)

def compute_logprob(clusters, cutoffs):
    # get log prob for each cluster
    cluster_logprobs = [tf.nn.log_softmax(cluster_logits) for cluster_logits in clusters]

    # adjust log prob for child clusters to account for probability
    # of picking child
    head_cluster_logprob = cluster_logprobs[0]
    for i in range(len(cutoffs) - 1):
        child_idx = cutoffs[0] + i
        # the log probability of choosing the ith child
        child_logprob = head_cluster_logprob[..., child_idx]
        child_logprob = tf.expand_dims(child_logprob, axis=-1)
        # add log probability of choosing child to every
        # item in child cluster
        cluster_logprobs[i+1] = tf.add(cluster_logprobs[i+1], child_logprob)

    # combine everything into flattened log prob
    head_categories_logprob = head_cluster_logprob[..., 0:(cutoffs[0])]
    return tf.concat([head_categories_logprob] + cluster_logprobs[1:], axis=-1)


class DifferentiatedSoftmaxProduceLogits(Layer):
    def __init__(self,
                 number_categories,
                 cutoffs,
                 capacities=None,
                 kernel_initializer='glorot_uniform',
                 projection_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(DifferentiatedSoftmaxProduceLogits, self).__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        self.projection_initializer = projection_initializer
        self.bias_initializer = bias_initializer
        self.cutoffs = list(cutoffs)
        if self.cutoffs[-1] < number_categories:
            self.cutoffs.append(number_categories)
        self.capacities = capacities
        self.number_categories = number_categories

    def build(self, input_shapes):
        input_size = input_shapes[-1]
        if not self.capacities:
            self.capacities = build_default_capacities(input_size, self.cutoffs)

        cluster_weight_shapes = build_cluster_weight_shapes(self.capacities, self.cutoffs, hierarchical=False)
        cluster_projection_shapes = build_cluster_projection_shapes(input_size, self.capacities)

        self.cluster_kernels = []
        self.cluster_biases = []
        self.cluster_projections = []
        for i in range(len(cluster_weight_shapes)):
            cluster_kernel_shape = cluster_weight_shapes[i]
            cluster_kernel = self.add_weight(name='cluster_kernel_%s' % i,
                                             shape=cluster_kernel_shape,
                                             initializer=self.kernel_initializer,
                                             trainable=True)

            cluster_bias = self.add_weight(name='cluster_bias_%s' % i,
                                           shape=(cluster_kernel_shape[1],),
                                           initializer=self.bias_initializer,
                                           trainable=True)

            cluster_projection_shape = cluster_projection_shapes[i]
            if cluster_projection_shape is None:
                cluster_projection = None
            else:
                cluster_projection = self.add_weight(name='cluster_projection_%s' % i,
                                                     shape=cluster_projection_shape,
                                                     initializer=self.projection_initializer,
                                                     trainable=True)

            self.cluster_kernels.append(cluster_kernel)
            self.cluster_biases.append(cluster_bias)
            self.cluster_projections.append(cluster_projection)

        super(DifferentiatedSoftmaxProduceLogits, self).build(input_shapes)

    def call(self, inputs, mask=None):
        cluster_inputs = [inputs]*len(self.cutoffs)
        logits = compute_logits(self.cluster_projections, self.cluster_kernels, self.cluster_biases, cluster_inputs)
        return tf.concat(logits, axis=-1)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shapes):
        return (*input_shapes[:-1], self.number_categories)

class AdaptiveSoftmaxProduceLogits(Layer):
    """
    # Arguments
        number_categories: Positive integer, dimensionality of the output space
        cutoffs: List of integers, the cutoff for each cluster (eg. if
            `number_categories` is 100 and cuttoffs is `[10, 30]` then 3
            clusters will be created, a head cluster for the first 10 categories
            the second cluster for the next 20 categories, and a third cluster
            for the final 70 categories).
        capacities: List of integers, representing the size of the output
            embeddings for each cluster.
        use_bias: Boolean, whether to use a bias.
        kernel_initializer: Initializer for cluster matrices.
        projection_initializer: Initializer for child cluster projections.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Regularizer for cluster matrices.
        projection_regularizer: Regularizer for child cluster projections.
        bias_regularizer: Regularizer for bias.

    # Input shape
        Option 1:
            nD input tensor with shape `(samples, ..., features)`
            nD label tensor with shape `(samples, ..., 1)`
        Option 2:
            nD input tensor with shape `(samples, ..., features)`

        (1) is used for training and will cause an appropriate cross-entropy
        loss to be added to the model. The labels must be provided as an input
        to determine the appropriate dot products to compute.
        (2) is useful when doing inference as it does not require the labels be
        provided as input and will also not cause a loss to be added to the
        model.

    # Output shape
        List of nD tensors representing the logits for each cluster. For
            example, if cutoffs is set to `[5000, 7000]` and `number_categories`
            is set to 10000, then the output shapes will be:

                `(samples, ..., 5002)`
                `(samples, ..., 2000)`
                `(samples, ..., 3000)`

            These logits can be appropriately normalized and merged using either
            a `AdaptiveProb` or `AdaptiveLogProb` layer.
    """
    def __init__(self,
                 number_categories,
                 cutoffs,
                 capacities=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 projection_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 projection_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(AdaptiveSoftmaxProduceLogits, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.projection_initializer = initializers.get(projection_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.cutoffs = list(cutoffs)
        if self.cutoffs[-1] < number_categories:
            self.cutoffs.append(number_categories)
        self.capacities = capacities
        self.use_bias = use_bias

    def build(self, input_shapes):
        if isinstance(input_shapes, list):
            input_size = input_shapes[0][-1]
        else:
            input_size = input_shapes[-1]

        if not self.capacities:
            self.capacities = build_default_capacities(input_size, self.cutoffs)

        cluster_weight_shapes = build_cluster_weight_shapes(self.capacities, self.cutoffs)
        cluster_projection_shapes = build_cluster_projection_shapes(input_size, self.capacities)

        self.cluster_kernels = []
        self.cluster_biases = []
        self.cluster_projections = []
        for i in range(len(cluster_weight_shapes)):
            cluster_kernel_shape = cluster_weight_shapes[i]
            cluster_kernel = self.add_weight(name='cluster_kernel_%s' % i,
                                             shape=cluster_kernel_shape,
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)

            if self.use_bias:
                cluster_bias = self.add_weight(name='cluster_bias_%s' % i,
                                               shape=(cluster_kernel_shape[1],),
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               trainable=True)
            else:
                cluster_bias = None

            cluster_projection_shape = cluster_projection_shapes[i]
            if cluster_projection_shape is None:
                cluster_projection = None
            else:
                cluster_projection = self.add_weight(name='cluster_projection_%s' % i,
                                                     shape=cluster_projection_shape,
                                                     initializer=self.projection_initializer,
                                                     regularizer=self.projection_regularizer,
                                                     trainable=True)

            self.cluster_kernels.append(cluster_kernel)
            self.cluster_biases.append(cluster_bias)
            self.cluster_projections.append(cluster_projection)

        super(AdaptiveSoftmaxProduceLogits, self).build(input_shapes)

    def compute_loss(self, a_prev, labels, mask=None):
        return compute_adaptive_loss(self.cluster_projections,
                                     self.cluster_kernels,
                                     self.cluster_biases,
                                     a_prev,
                                     labels,
                                     self.cutoffs,
                                     mask=mask)

    def call(self, inputs, mask=None):
        a, labels = None, None

        if isinstance(inputs, list):
            if len(inputs) == 1:
                a = inputs[0]
            elif len(inputs) == 2:
                a, labels = inputs
            else:
                raise ValueError('Expected 1 or 2 inputs but received %s.' % len(inputs))
        else:
            a = inputs

        if labels is not None:
            a_masking = mask[0] if mask else None
            labels = tf.squeeze(labels, axis=-1)
            self.add_loss(self.compute_loss(a, labels, mask=a_masking), inputs)

        cluster_inputs = [a] * len(self.cutoffs)
        logits = compute_logits(self.cluster_projections,
                                self.cluster_kernels,
                                self.cluster_biases,
                                cluster_inputs)
        return logits

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return [mask] * len(self.cutoffs)

    def compute_output_shape(self, input_shapes):
        a_shape, lables_shape = None, None

        if isinstance(input_shapes, list):
            if len(input_shapes) == 1:
                a_shape = input_shapes[0]
            elif len(input_shapes) == 2:
                a_shape, lables_shape = input_shapes
            else:
                raise ValueError('Expected 1 or 2 inputs but received %s.' % len(input_shapes))
        else:
            a_shape = input_shapes

        m = a_shape[0]
        max_capacity = a_shape[-1]
        cluster_weight_shapes = build_cluster_weight_shapes(self.capacities, self.cutoffs)
        output_shapes = []
        for i in range(len(cluster_weight_shapes)):
            (capacity, output_dim) = cluster_weight_shapes[i]
            output_shapes.append((m, *(a_shape[1:-1]), output_dim))

        return output_shapes

class AdaptiveProb(Layer):
    def __iniit__(self, **kwargs):
        super(AdaptiveProb, self).__init__(**kwargs)

    def build(self, input_shapes):
        cutoffs = []
        number_clusters = len(input_shapes)
        number_children = number_clusters - 1
        cutoffs = [shape[-1] for shape in input_shapes]
        cutoffs[0] = cutoffs[0] - number_children
        self.cutoffs = cutoffs
        super(AdaptiveProb, self).build(input_shapes)

    def call(self, inputs, mask=None):
        return compute_prob(inputs, self.cutoffs)

    def compute_mask(self, inputs, mask=None):
        if mask:
            return mask[0]
        return mask

class AdaptiveLogProb(Layer):
    def __iniit__(self, **kwargs):
        super(AdaptiveLogProb, self).__init__(**kwargs)

    def build(self, input_shapes):
        cutoffs = []
        number_clusters = len(input_shapes)
        number_children = number_clusters - 1
        cutoffs = [shape[-1] for shape in input_shapes]
        cutoffs[0] = cutoffs[0] - number_children
        self.cutoffs = cutoffs
        super(AdaptiveLogProb, self).build(input_shapes)

    def call(self, inputs, mask=None):
        return compute_logprob(inputs, self.cutoffs)

    def compute_mask(self, inputs, mask=None):
        if mask:
            return mask[0]
        return mask
