"""Keras Tutorials

This script contains tutorials from the book: Deep Learning with Python - François Chollet
Each function is meant to be a temporary "main" function.
"""

import logging
import os
import time

from keras.datasets import imdb
from keras.layers import Embedding
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
import numpy as np

from usermodeling import utils

# Change the level of the loggers of some of the imported modules
logging.getLogger("matplotlib").setLevel(logging.INFO)


def tutorial_onehot_encoding():
    """One-hot encoding

    Listing 6.3  Using Keras for word-level one-hot encoding
    Page 183 of the book: Deep Learning with Python - François Chollet
    """

    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    # Create a tokenizer, configured to only take into account the 1000 most common words
    tokenizer = Tokenizer(num_words=1000)

    # Turn strings into lists of integer indices
    tokenizer.fit_on_texts(samples)

    sequences = tokenizer.texts_to_sequences(samples)
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    word_index = tokenizer.word_index


def tutorial_word_embeddings():
    """Word embeddings

    Listing 6.6  Loading the IMDB data for use with an Embedding layer
    Listing 6.7  Using an Embedding layer and classifier on the IMDB data

    Page 187 of the book: Deep Learning with Python - François Chollet
    """
    # Number of words to consider as features
    max_features = 10000

    # Cut off the text after this many words (among the max_features most common words)
    maxlen = 20

    # Load the data as lists of integers
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # Turn the lists of integers into a 2D integer tensor of shape (samples, maxlen)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    # Embedding layer. After the Embedding layer, the  activations have shape (samples, maxlen, 8)
    model.add(Embedding(10000, 8, input_length=maxlen))
    # Flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, maxlen * 8)
    model.add(Flatten())
    # Add the classifier on top
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary(print_fn=logger.info)

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


def tutorial_pretrained_word_embeddings():
    """Using pre-trained word embeddings

    Listing 6.8  Processing the labels of the raw IMDB data
    Listing 6.9  Tokenizing the text of the raw IMDB data
    Listing 6.10  Parsing the GloVe word-embeddings file
    Listing 6.11  Preparing the GloVe word-embeddinggs matrix
    Listing 6.12  Model definition
    Listing 6.13  Loading pretrained word embeddings into the Embedding layer
    Listing 6.14  Training and evaluation
    Listing 6.15  Plotting the results
    Listing 6.16
    Listing 6.17  Tokenizing the data of the test set
    Listing 6.18  Evaluating the model on the test set

    Page 189–195 of the book: Deep Learning with Python - François Chollet

    Remarks:
    - First, download the raw IMDB dataset from http://mng.bz/0tIo (https://s3.amazonaws.com/text-datasets/aclImdb.zip)
      and uncompress it.
    """

    # Listing 6.8  Processing the labels of the raw IMDB data
    IMDB_DIR = 'data/IMDB - Keras Tutorial/aclImdb'
    TRAIN_DIR = os.path.join(IMDB_DIR, 'train')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(TRAIN_DIR, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding='utf-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    logger.info('@ %.2f seconds: Finished processing the labels of the raw IMDB data', time.process_time())


    # Listing 6.9  Tokenizing the text of the raw IMDB data
    MAXLEN = 100  # Cut off reviews after this many words
    TRAINING_SAMPLES = 20000  # Value in the tutorial: 200
    VALIDATION_SAMPLES = 5000  # Value in the tutorial: 10,000
    MAX_WORDS = 10000  # Consider only the top 10,000 words in the dataset

    logger.info("Training set: %d samples  |  Validation set: %d samples", TRAINING_SAMPLES, VALIDATION_SAMPLES)
    logger.info("MAXLEN = %d  |  MAX_WORDS = %d", MAXLEN, MAX_WORDS)

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)  # Builds the word index
    sequences = tokenizer.texts_to_sequences(texts)  # Turns the strings into lists of integer indices
    word_index = tokenizer.word_index  # How you can recover the word index that was computed
    logger.info('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAXLEN)

    labels = np.asarray(labels)
    logger.info('Shape of data tensor: %s', data.shape)
    logger.info('Shape of label tensor: %s', labels.shape)

    # Shuffle the data and the labels
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Split the data into a training set and a validation set
    x_train = data[:TRAINING_SAMPLES]
    y_train = labels[:TRAINING_SAMPLES]
    x_val = data[TRAINING_SAMPLES: TRAINING_SAMPLES + VALIDATION_SAMPLES]
    y_val = labels[TRAINING_SAMPLES: TRAINING_SAMPLES + VALIDATION_SAMPLES]

    logger.info('@ %.2f seconds: Finished tokenizing the text of the raw IMDB data', time.process_time())


    # Listing 6.10  Parsing the GloVe word-embeddings file
    GLOVE_DIR = 'data/GloVe - Twitter Word Embeddings'
    GLOVE_PATH = os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt')

    embeddings_index = {}  # Create an empty dictionary

    with open(GLOVE_PATH, 'r', encoding='utf-8') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            # ↳ Note that index 0 is not supposed to stand for any word or token—it's a placeholder.

    logger.info('Found %s word vectors.', len(embeddings_index))
    logger.info('@ %.2f seconds: Finished parsing the GloVe word-embeddings file', time.process_time())


    # Listing 6.11  Preparing the GloVe word-embeddinggs matrix
    EMBEDDING_DIM = 100

    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < MAX_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                # Words not found in the embedding index will be all zeros.

    logger.info('@ %.2f seconds: Finished preparing the GloVe word-embeddinggs matrix', time.process_time())


    # L)isting 6.12  Model definition
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAXLEN))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=logger.info)


    # Listing 6.13  Loading pretrained word embeddings into the Embedding layer
    # Comment this section ⇒ Listing 6.16
    model.layers[0].set_weights([embedding_matrix])
    # Freeze the Embedding layer
    model.layers[0].trainable = False


    # Listing 6.14  Training and evaluation
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        )
    model.save_weights('data/out/pre_trained_glove_model.h5')

    logger.info('@ %.2f seconds: Finished training and evaluation', time.process_time())


    # Listing 6.15  Plotting the results
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    logger.info('Training acc: %s', acc)
    logger.info('Validation acc: %s', val_acc)
    logger.info('Training loss: %s', loss)
    logger.info('Validation loss: %s', val_loss)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    # Create a new figure
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


    # Listing 6.17  Tokenizing the data of the test set
    TEST_DIR = os.path.join(IMDB_DIR, 'test')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(TEST_DIR, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    sequences = tokenizer.texts_to_sequences(texts)
    x_test = pad_sequences(sequences, maxlen=MAXLEN)
    y_test = np.asarray(labels)

    logger.info('@ %.2f seconds: Finished tokenizing the data of the test set', time.process_time())


    # Listing 6.18  Evaluating the model on the test set
    model.load_weights('data/out/pre_trained_glove_model.h5')
    metrics_values = model.evaluate(x_test, y_test)
    logger.info('@ %.2f seconds: Finished evaluating the model on the test set', time.process_time())
    for name, value in zip(model.metrics_names, metrics_values):
        logger.info("%s: %s", name, value)


    # logger.info('TEMP!')


def tutorial_template():
    """Tutorial

    Listing #.#  Title
    Page ### of the book: Deep Learning with Python - François Chollet
    """

    # Start here


def main():
    """The main function.

    Every time the script runs, it will call this function.
    """

    # Log run time
    logger.info('@ %.2f seconds: Run finished', time.process_time())


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to '__main__' when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == '__main__':
    logger = utils.configure_root_logger()
    utils.set_working_directory()
    # tutorial_onehot_encoding()
    # tutorial_word_embeddings()
    tutorial_pretrained_word_embeddings()
    main()
