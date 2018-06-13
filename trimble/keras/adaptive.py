import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

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
        child_labels = tf.expand_dims(child_labels, 1)
    #                 labels_child = tf.Print(labels_child, [labels_child], message="labels %s: " % (i+1), summarize=10)
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
    head_labels = K.reshape(head_labels, K.constant([-1, 1], dtype='int32'))

    cluster_labels.insert(0, head_labels)
    return cluster_labels

def compute_cluster_inputs(inputs, child_cluster_masks, cutoffs, mask=None):
    # input for this cluster
    has_steps = len(K.int_shape(inputs)) > 2

    child_inputs = []
    for i in range(len(child_cluster_masks)):
        child_cluster_mask = child_cluster_masks[i]

        if not has_steps:
            child_cluster_mask = tf.squeeze(child_cluster_mask, axis=[-1])

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
        x = K.bias_add(x, bias)
#             x = tf.Print(x, [tf.shape(x)])
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
    batch_size = tf.shape(a_prev)[0]
    has_steps = len(K.int_shape(logits[0])) > 2

    total_cost = None
    for logits, labels in zip(logits, cluster_labels):
        if not has_steps:
            labels = tf.squeeze(labels, axis=1)
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cost = tf.reduce_sum(cost)
        if total_cost is None:
            total_cost = cost
        else:
            total_cost = tf.add(total_cost, cost)

    cost_mean = tf.divide(total_cost, tf.cast(batch_size, 'float32'))
    return cost_mean

def compute_softmax(clusters, cutoffs):
    cluster_softmaxes = [K.softmax(cluster_logits) for cluster_logits in clusters]

    head_cluster_softmax = cluster_softmaxes[0]
    for i in range(len(cutoffs) - 1):
        child_idx = cutoffs[0]

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
        return (input_shapes[0], self.number_categories)

class AdaptiveSoftmaxProduceLogits(Layer):
    def __init__(self,
                 number_categories,
                 cutoffs,
                 capacities=None,
                 kernel_initializer='glorot_uniform',
                 projection_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(AdaptiveSoftmaxProduceLogits, self).__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        self.projection_initializer = projection_initializer
        self.bias_initializer = bias_initializer
        self.cutoffs = list(cutoffs)
        if self.cutoffs[-1] < number_categories:
            self.cutoffs.append(number_categories)
        self.capacities = capacities

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
