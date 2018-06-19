"""
Trains a simple language model based on the one found at
https://github.com/facebookresearch/adaptive-softmax and generates comparative
results for full softmax, differentiated softmax, and adaptive softmax. This
benchmark uses the text8 (http://mattmahoney.net/dc/textdata.html) dataset. This
dataset isn't the best demonstration of adaptive softmax's strengths, but it is
of a convenient size for downloading and training in a reasonable amount of
time.

You can run the benchmark by executing the following at the project root:

    PYTHONPATH="$PYTHONPATH:." python examples/text8_benchmark.py --graph

You can see all of the other options by using the `--help` option:

    usage: text8_benchmark.py [-h] [-b {adaptive,full,differentiated}]
                          [--no-resume] [--output-directory OUTPUT_DIRECTORY]
                          [--graph]

    optional arguments:
      -h, --help            show this help message and exit
      -b {adaptive,full,differentiated}, --benchmarks {adaptive,full,differentiated}
                            run benchmark for different variations of softmax
      --no-resume           prevents resuming a previously interrupted benchmark
      --output-directory OUTPUT_DIRECTORY
                            where to store output of benchmark
      --graph               dump a graph of perplexity over time for bencmarks

By default, the benchmark runs for every variation of softmax. This can take a
long time to train on the CPU (over a day) so use of a GPU is recommended.
"""
from keras.utils.data_utils import get_file
from keras.preprocessing import text
from keras.preprocessing import sequence

from keras.models import Model
from keras.layers import (Dense,
                          Dropout,
                          Input,
                          LSTM,
                          Embedding)
from keras.optimizers import Adagrad
from trimble.keras.adaptive import (DifferentiatedSoftmaxProduceLogits,
                                    AdaptiveSoftmaxProduceLogits,
                                    AdaptiveLogProb)

from zipfile import ZipFile

import numpy as np
import tensorflow as tf
import math
import io
import time
import os
import json

TEXT8_DATA_URL='http://mattmahoney.net/dc/text8.zip'

def segment_sequence_into_batches(data, batch_size=128, sequence_length=20):
    """
    For performance reasons, it is often desirable to train a language model on
    batches of multiple sequences of text at a time as opposed to on one
    continuous sequence of text. This function takes a continuous sequence of
    data and reshapes such that when passed to a function like `Model.fit`, the
    samples of batch `n` will be a continuation of the samples of batch `n-1`,
    and the samples of batch `n-1` will be a continuation of the samples of
    batch `n-2`, etc. This will permit RNN state to be carried over productively
    from batch to batch.

    Consider a sequence like the following:

        >>> data = list(range(1,14))
        >>> data
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    calling this function on the data provides the following:

        >>> X = segment_sequence_into_batches(data, batch_size=3, sequence_length=4)
        >>> X
        array([[ 1,  2,  3,  4],
               [ 6,  7,  8,  9],
               [11, 12, 13,  0],
               [ 5,  0,  0,  0],
               [10,  0,  0,  0],
               [ 0,  0,  0,  0]], dtype=int32)

    if we look at the first batch:

        >>> X[0:3]
        array([[ 1,  2,  3,  4],
               [ 6,  7,  8,  9],
               [11, 12, 13,  0]], dtype=int32)

    and the second batch:

        >>> X[3:6]
        array([[ 5,  0,  0,  0],
               [10,  0,  0,  0],
               [ 0,  0,  0,  0]], dtype=int32)

    we can see that the samples from the second batch are a continuation of
    those from the first.
    """
    segment_length = math.ceil(len(data) / batch_size)
    total_batches = math.ceil(segment_length / sequence_length)
    x = []
    for batch_idx in range(total_batches):
        for segment_idx in range(batch_size):
            segment_start = segment_idx*segment_length
            segment_end = (segment_idx+1)*segment_length
            batch_start = batch_idx*sequence_length
            batch_end = (batch_idx+1)*sequence_length
            sample_start = min(segment_start+batch_start, segment_end)
            sample_end = min(segment_start+batch_end, segment_end)
            sample = data[sample_start:sample_end]
            x.append(sample)
    return sequence.pad_sequences(x, padding='post', maxlen=sequence_length)

def _build_tokenizer(data, vocab_size=45000):
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts([data])
    word_index = {w: i for (w,i) in tokenizer.word_index.items() if i < vocab_size - 1}
    tokenizer = text.Tokenizer(num_words=vocab_size, oov_token=vocab_size-1)
    tokenizer.word_index = word_index
    return tokenizer

def _load_raw_text8_data():
    dirname = 'text8.zip'
    path = get_file(
        dirname,
        origin=TEXT8_DATA_URL,
        md5_hash='f26f94c5209bc6159618bad4a559ff81',
        archive_format='zip')

    with ZipFile(path) as text8zip:
        with io.TextIOWrapper(text8zip.open('text8'), encoding='utf-8') as text8file:
            return text8file.read()

def load_data(vocab_size=45000, batch_size=128, sequence_length=20):
    """Loads Text8 dataset. (http://mattmahoney.net/dc/textdata.html)

    # Arguments
        vocab_size: maximum number of words to use.
        batch_size: the batch size that will be used when this data is passed to
            `Model.fit(..)` or similar function.
        sequence_length: the number of time steps for each batch.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    raw_data = _load_raw_text8_data()

    tokenizer = _build_tokenizer(raw_data, vocab_size=vocab_size)

    train_text = raw_data[:99000000]
    dev_text = raw_data[99000000:]

    raw_data = None # allow gc

    result = []
    for data_text in [train_text, dev_text]:
        data_sequence = tokenizer.texts_to_sequences([data_text])[0]
        data_input = segment_sequence_into_batches(
            data_sequence,
            batch_size=batch_size,
            sequence_length=sequence_length)
        data_labels = segment_sequence_into_batches(
            data_sequence[1:] + [0],
            batch_size=batch_size,
            sequence_length=sequence_length)
        result.append((data_input, data_labels))

    return tuple(result)

def get_word_index(vocab_size=45000):
    raw_data = _load_raw_text8_data()
    tokenizer = _build_tokenizer(raw_data, vocab_size=vocab_size)
    return tokenizer.word_index

def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    return loss

def _build_base_model(
        batch_size=128,
        sequence_length=20,
        vocab_size=45000,
        input_word_vectors_dim=256,
        hidden_dim=2048,
        dropout=0.0):

    inputs = Input(name='data_input', batch_shape=(batch_size, sequence_length), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=input_word_vectors_dim, mask_zero=True)
    dropout_pre_rnn = Dropout(dropout)
    rnn = LSTM(hidden_dim, return_sequences=True, stateful=True)
    dropout_post_rnn = Dropout(dropout)

    x = inputs
    x = embedding(x)
    x = dropout_pre_rnn(x)
    x = rnn(x)
    x = dropout_post_rnn(x)

    return (inputs, x)

def build_adaptive_softmax_model(
        cutoffs,
        batch_size=128,
        sequence_length=20,
        vocab_size=45000,
        input_word_vectors_dim=256,
        hidden_dim=2048,
        dropout=0.0,
        lr=0.1,
        clipnorm=1):
    labels = Input(name='labels_input', batch_shape=(batch_size, sequence_length), dtype='int32')

    (inputs, x) = _build_base_model(batch_size=batch_size,
                                    sequence_length=sequence_length,
                                    vocab_size=vocab_size,
                                    input_word_vectors_dim=input_word_vectors_dim,
                                    hidden_dim=hidden_dim,
                                    dropout=dropout)
    adaptive_softmax_layer = AdaptiveSoftmaxProduceLogits(vocab_size, cutoffs=cutoffs)
    x = adaptive_softmax_layer([x, labels])

    model = Model(inputs=[inputs, labels], outputs=x)
    optimizer = Adagrad(lr=lr, clipnorm=clipnorm)
    model.compile(optimizer=optimizer)

    return model

def build_full_softmax_model(
        batch_size=128,
        sequence_length=20,
        vocab_size=45000,
        input_word_vectors_dim=256,
        hidden_dim=2048,
        dropout=0.0,
        lr=0.1,
        clipnorm=1):

    (inputs, x) = _build_base_model(batch_size=batch_size,
                                    sequence_length=sequence_length,
                                    vocab_size=vocab_size,
                                    input_word_vectors_dim=input_word_vectors_dim,
                                    hidden_dim=hidden_dim,
                                    dropout=dropout)

    x = Dense(vocab_size)(x)
    model = Model(inputs=inputs, outputs=x)
    optimizer = Adagrad(lr=lr, clipnorm=clipnorm)
    # This bit of hoodoo is thanks to https://github.com/tensorflow/tensorflow/issues/17150. It works around
    # a bug where Keras cannot figure out the proper output shape.
    dummy_target = tf.placeholder(dtype='int32', shape=(None, None))
    model.compile(optimizer=optimizer, loss=sparse_cross_entropy, target_tensors=[dummy_target])

    return model

def build_differentiated_softmax_model(
        cutoffs,
        batch_size=128,
        sequence_length=20,
        vocab_size=45000,
        input_word_vectors_dim=256,
        hidden_dim=2048,
        dropout=0.0,
        lr=0.1,
        clipnorm=1):

    (inputs, x) = _build_base_model(batch_size=batch_size,
                                    sequence_length=sequence_length,
                                    vocab_size=vocab_size,
                                    input_word_vectors_dim=input_word_vectors_dim,
                                    hidden_dim=hidden_dim,
                                    dropout=dropout)

    x = DifferentiatedSoftmaxProduceLogits(vocab_size, cutoffs)(x)
    model = Model(inputs=inputs, outputs=x)
    optimizer = Adagrad(lr=lr, clipnorm=clipnorm)
    # This bit of hoodoo is thanks to https://github.com/tensorflow/tensorflow/issues/17150. It works around
    # a bug where Keras cannot figure out the proper output shape.
    dummy_target = tf.placeholder(dtype='int32', shape=(None, None))
    model.compile(optimizer=optimizer, loss=sparse_cross_entropy, target_tensors=[dummy_target])

    return model

def train_model(model, epochs, train_data, validation_data, batch_size=128, labels_as_input=False):
    (x_train, y_train) = train_data
    (x_valid, y_valid) = validation_data

    model.reset_states()

    times = [0]
    ppls = []

    # get a measurement before any training so that we having something
    # for time 0

    if labels_as_input:
        valid_loss = model.evaluate(x=[x_valid, y_valid], batch_size=128)
    else:
        valid_loss = model.evaluate(x=x_valid, y=y_valid, batch_size=128)

    ppls.append(math.e ** valid_loss)

    for epoch in range(epochs):
        start_time = time.time()

        model.reset_states()

        if labels_as_input:
            model.fit(x=[x_train, y_train], batch_size=128, epochs=epoch+1, initial_epoch=epoch, shuffle=False)
        else:
            model.fit(x=x_train, y=y_train, batch_size=128, epochs=epoch+1, initial_epoch=epoch, shuffle=False)


        end_time = time.time()
        times.append(end_time - start_time)

        model.reset_states()
        valid_loss = None

        try:
            if labels_as_input:
                valid_loss = model.evaluate(x=[x_valid, y_valid], batch_size=128)
            else:
                valid_loss = model.evaluate(x=x_valid, y=y_valid, batch_size=128)
        finally:
            model.reset_states()
        ppl = math.e ** valid_loss
        print("ppl: %s" % ppl)
        ppls.append(ppl)

    return times, ppls

def format_duration(seconds):
    hours = seconds // (60**2)
    seconds = seconds - hours*(60**2)
    minutes = seconds // 60
    seconds = seconds - minutes*60
    return "%02d:%02d:%06.3f" % (hours, minutes, seconds)

def print_summary(benchmark_results):
    for (label, [times, ppls]) in benchmark_results:
        total_time = sum(times)
        epochs = len(times) - 1
        ppl = ppls[-1]
        print()
        print(label)
        print("Epochs: %s" % epochs)
        print("Training time: %s" % format_duration(total_time))
        print("Perplexity: %.2f" % ppl)
        print()

def dump_graph(results, destination_path):
    import pprint
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = []
    for (label, [times, ppls]) in results:
        cumulative_times = [sum(times[:i]) for i in range(1,len(times)+1)]
        data.append((label, [cumulative_times, ppls]))

    shapes = ['o', '^', 's']
    colors = ['b', 'g', 'r', 'y']

    for i in range(len(data)):
        (label, [times, ppls]) = data[i]
        times = [t / 60 for t in times]
        color = colors[i % len(colors)]
        shape = shapes[i % len(shapes)]
        plt.plot(times[1:], ppls[1:], "%s-" % color, label=label)
        plt.plot(times[1:], ppls[1:], "%s%s" % (color, shape), label=label)
    plt.ylim([80, 600])
    plt.ylabel('Perplexity')
    plt.xlim(xmin=0)
    plt.xlabel('Time (minutes)')
    plt.legend()
    plt.savefig(destination_path)

def run_benchmarks(epochs, benchmarks=None, output_directory='benchmark_out', resume=False):
    (x_train, y_train), (x_valid, y_valid) = load_data()

    model_options = {'lr': 0.1, 'dropout': 0.25, 'hidden_dim': 512, 'input_word_vectors_dim': 512, 'clipnorm': 1}

    def full_softmax_benchmark():
        model = build_full_softmax_model(**model_options)
        return model, train_model(model, epochs, (x_train, y_train), (x_valid, y_valid), labels_as_input=False)

    def adaptive_softmax_benchmark():
        model = build_adaptive_softmax_model([2000,10000], **model_options)
        return model, train_model(model, epochs, (x_train, y_train), (x_valid, y_valid), labels_as_input=True)

    def differentiated_softmax_benchmark():
        model = build_differentiated_softmax_model([2000,10000], **model_options)
        return model, train_model(model, epochs, (x_train, y_train), (x_valid, y_valid), labels_as_input=False)

    benchmark_descriptors = [
        ('full', 'Full Softmax', full_softmax_benchmark),
        ('adaptive', 'Adaptive Softmax', adaptive_softmax_benchmark),
        ('differentiated', 'Differentiated Softmax', differentiated_softmax_benchmark)
    ]

    if benchmarks is None:
        benchmarks = [benchmark_id for (benchmark_id, *_) in benchmark_descriptors]

    results = []

    for benchmark_id, label, benchmark_fn in benchmark_descriptors:
        if not benchmark_id in benchmarks:
            continue

        filename = os.path.join(output_directory, benchmark_id)
        stats_filename = filename + '.json'
        weights_filename = filename + '.h5'

        if resume and os.path.exists(stats_filename) and os.path.exists(weights_filename):
            with open(stats_filename) as f:
                results.append((label, json.load(f)))
        else:
            model, stats = benchmark_fn()
            model.save_weights(weights_filename)
            with open(stats_filename, 'w') as f:
                json.dump(stats, f)
            results.append((label, stats))

    return results

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--benchmarks',
                        choices=['adaptive', 'full', 'differentiated'],
                        action='append',
                        help="run benchmark for different variations of softmax")

    parser.add_argument('--no-resume',
                        dest='resume',
                        action='store_false',
                        help="prevents resuming a previously interrupted benchmark")

    parser.add_argument('--output-directory',
                        dest="output_directory",
                        default='benchmark_out',
                        help="where to store output of benchmark")

    parser.add_argument('--graph',
                        action='store_true',
                        help="dump a graph of perplexity over time for bencmarks")

    options = parser.parse_args()
    options.benchmarks = options.benchmarks or ['adaptive', 'full', 'differentiated']

    if not os.path.exists(options.output_directory):
        os.mkdir(options.output_directory)

    result = run_benchmarks(10, benchmarks=options.benchmarks, output_directory=options.output_directory, resume=options.resume)
    print_summary(result)
    if options.graph:
        dump_graph(result, os.path.join(options.output_directory, 'text8_performance_comparison.png'))
