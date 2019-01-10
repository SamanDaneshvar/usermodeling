"""Keras Tutorials

This script contains tutorials from the book: Deep Learning with Python - François Chollet
Each function is meant to be a temporary "main" function.
"""

from usermodeling import utils

import time

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense


def tutorial_onehot_encoding():
    """One-hot encoding

    Listing 6.3 Using Keras for word-level one-hot encoding
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

    Listing 6.6 Loading the IMDB data for use with an Embedding layer
    Listing 6.7 Using an Embedding layer and classifier on the IMDB data
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
    model.summary()

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


def tutorial_pretrained_word_embeddings():
    """Using pre-trained word embeddings

    Listing 6.8 Processing the labels of the raw IMDB data
    Listing ### ...
    Page 189–### of the book: Deep Learning with Python - François Chollet
    """

    # Listing 6.8 Processing the labels of the raw IMDB data
    imdb_dir = ''


def tutorial_template():
    """Tutorial

    Listing #.# Title
    Page ### of the book: Deep Learning with Python - François Chollet
    """

    # Start here


def main():
    """The main function.

    Every time the script runs, it will call this function.
    """

    # Log run time
    logger.info("@ %.2f seconds: Run finished\n", time.process_time())


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to "__main__" when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    logger = utils.configure_root_logger()
    utils.set_working_directory()
    # tutorial_onehot_encoding()
    # tutorial_word_embeddings()
    # tutorial_pretrained_word_embeddings()
    main()
