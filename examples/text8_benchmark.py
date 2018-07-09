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
from keras import initializers
from keras.models import Model
from keras.layers import (Dense,
                          Dropout,
                          Input,
                          LSTM,
                          Embedding,
                          Activation)
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

def _load_raw_text8_data(output_directory='./benchmark_out'):
    dirname = 'text8.zip'
    path = get_file(
        dirname,
        origin=TEXT8_DATA_URL,
        md5_hash='f26f94c5209bc6159618bad4a559ff81',
        archive_format='zip',
        cache_dir=output_directory)

    with ZipFile(path) as text8zip:
        with io.TextIOWrapper(text8zip.open('text8'), encoding='utf-8') as text8file:
            return text8file.read()

def _build_tokenizer(data, vocab_size=45000):
    num_words = vocab_size-1
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts([data])
    words = ['</s>', '<unk>']
    words.extend([w for (w,_) in sorted(list(tokenizer.word_index.items()), key=lambda x: x[1])])
    words = words[:num_words]
    word_index = dict(zip(words, range(1, vocab_size)))
    tokenizer = text.Tokenizer(num_words=min(num_words, len(word_index)), oov_token='<unk>')
    tokenizer.word_index = word_index
    return tokenizer

def _split_text8(text8_text):
    return text8_text[:99000000], text8_text[99000000:]

def _segment_sequence_into_batches(data_sequence, eos_idx, batch_size, sequence_length):
    number_batches = int(np.ceil(len(data_sequence) / (batch_size*sequence_length)))
    data = np.full(number_batches*sequence_length*batch_size, eos_idx, dtype='int32')
    data[-len(data_sequence):] = data_sequence
    data = np.reshape(data, (batch_size, -1))

    x = np.roll(data, 1, axis=1)
    x[:, 0] = eos_idx
    x = np.vstack(np.hsplit(x, x.shape[1] // 20))

    labels = np.vstack(np.hsplit(data, data.shape[1] // 20))
    labels = np.expand_dims(labels, axis=-1)
    return x, labels

def load_data(vocab_size=45000, batch_size=128, sequence_length=20, output_directory='./benchmark_out'):
    """
    Loads Text8 dataset. (http://mattmahoney.net/dc/textdata.html)

    # Arguments
        vocab_size: maximum number of words to use.
        batch_size: the batch size that will be used when this data is passed to
            `Model.fit(..)` or similar function.
        sequence_length: the number of time steps for each batch.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    raw_data = _load_raw_text8_data(output_directory=output_directory)
    train_text, dev_text = _split_text8(raw_data)
    tokenizer = _build_tokenizer(train_text, vocab_size=vocab_size)
    raw_data = None # allow gc
    eos_idx = tokenizer.word_index['</s>']

    results = []
    data_sequence = tokenizer.texts_to_sequences([train_text])[0] + [eos_idx]
    results.append(_segment_sequence_into_batches(data_sequence, eos_idx, batch_size, sequence_length))
    data_sequence = tokenizer.texts_to_sequences([dev_text])[0] + [eos_idx]
    results.append(_segment_sequence_into_batches(data_sequence, eos_idx, batch_size, sequence_length))

    return tuple(results)

def get_word_index(vocab_size=45000, output_directory="./benchmark_out"):
    raw_data = _load_raw_text8_data(output_directory=output_directory)
    train_text, _ = _split_text8(raw_data)
    tokenizer = _build_tokenizer(train_text, vocab_size=vocab_size)
    return tokenizer.word_index

def sparse_cross_entropy(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
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
    embedding = Embedding(input_dim=vocab_size, output_dim=input_word_vectors_dim, mask_zero=False)
    dropout_pre_rnn = Dropout(dropout)
    rnn = LSTM(hidden_dim, return_sequences=True, stateful=True, unroll=True)
    dropout_post_rnn = Dropout(dropout)

    x = inputs
    x = embedding(x)
    x = dropout_pre_rnn(x)
    x = rnn(x)
    x = dropout_post_rnn(x)

    return (inputs, x)

def build_adaptive_softmax_model(cutoffs, lr=0.1, batch_size=128, sequence_length=20, vocab_size=45000, **kwargs):
    labels = Input(name='labels_input', batch_shape=(batch_size, sequence_length, 1), dtype='int32')

    (inputs, x) = _build_base_model(batch_size=batch_size, sequence_length=sequence_length, vocab_size=vocab_size, **kwargs)
    adaptive_softmax_layer = AdaptiveSoftmaxProduceLogits(vocab_size, cutoffs=cutoffs)
    x = adaptive_softmax_layer([x, labels])

    model = Model(inputs=[inputs, labels], outputs=x)
    optimizer = Adagrad(lr=lr)
    model.compile(optimizer=optimizer)

    return model

def build_full_softmax_model(lr=0.1, vocab_size=45000, **kwargs):
    (inputs, x) = _build_base_model(**kwargs)
    x = Dense(vocab_size, activation='linear')(x)

    model = Model(inputs=inputs, outputs=x)
    optimizer = Adagrad(lr=lr)
    model.compile(optimizer=optimizer, loss=sparse_cross_entropy)

    return model

def build_differentiated_softmax_model(cutoffs, lr=0.1, vocab_size=45000, **kwargs):
    (inputs, x) = _build_base_model(vocab_size=vocab_size, **kwargs)
    x = DifferentiatedSoftmaxProduceLogits(vocab_size, cutoffs)(x)

    model = Model(inputs=inputs, outputs=x)
    optimizer = Adagrad(lr=lr)
    model.compile(optimizer=optimizer, loss=sparse_cross_entropy)

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
        plt.plot(times[1:], ppls[1:], "%s%s" % (color, shape))
    plt.ylim([80, 600])
    plt.ylabel('Perplexity')
    plt.xlim(xmin=0)
    plt.xlabel('Time (minutes)')
    plt.legend()
    plt.savefig(destination_path)

def run_benchmarks(epochs, benchmarks=None, output_directory='benchmark_out', resume=False):
    (x_train, y_train), (x_valid, y_valid) = load_data(output_directory=output_directory)

    model_options = {'lr': 0.1, 'dropout': 0.25, 'hidden_dim': 512, 'input_word_vectors_dim': 512}

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

    parser.add_argument('--iterations',
                        type=int,
                        default=10,
                        help="number of training iterations")

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

    result = run_benchmarks(options.iterations, benchmarks=options.benchmarks, output_directory=options.output_directory, resume=options.resume)
    print_summary(result)
    if options.graph:
        dump_graph(result, os.path.join(options.output_directory, 'text8_performance_comparison.png'))
