# Fix the SyntaxError (http://python.org/dev/peps/pep-0263/) on Compute Canada's login node
# coding=utf-8

"""Perform deep learning experiments on the datasets

This script trains a deep learning model on the datasets.
"""

import logging
import os
import pickle
import random as rn
import time

from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from usermodeling import datasets
from usermodeling import def_train_model
from usermodeling.pan18ap_classical_ml import preprocess_tweet
from usermodeling.utils import my_utils

# Change the level of the loggers of some of the imported modules
logging.getLogger("matplotlib").setLevel(logging.INFO)


def ensure_reproducibility():
    """Ensure reproducible results

    Ensure the reproducibility of the experiments by seeding the pseudo-random number generators and some other
    TensorFlow and Keras session configurations.

    Usage: Run this function before building your model. Moreover, run the *keras.backend.clear_session()* function
    after your experiment to restart with a fresh session.

    - Note that the most robust way to report results and compare models is to repeat your experiment many times (30+)
    and use summary statistics.

    References:
        https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
        https://stackoverflow.com/a/46886311/9933071
    """

    # Set the random seed for NumPy. Keras gets its source of randomness from NumPy.
    np.random.seed(42)
    # Set the random seed for the TensorFlow backend.
    tf.set_random_seed(123)

    # Set the random seed for the core Python random number generator.
    # Not sure of the effectiveness of this, but it is recommended by Keras documentation.
    rn.seed(1234)

    # Force TensorFlow to use a single thread. Multiple threads are a potential source of non-reproducible results.
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def load_split_and_vectorize_pan18ap_data(MAX_WORDS, MAX_SEQUENCE_LEN):
    """Load, vectorize and split the PAN 2018 data

    - Load the English training dataset of the PAN 2018 Author Profiling task
    - Pre-process the raw text (replace URLs, etc.)
    - Split the dataset into balanced (stratified) training (60%), validation (20%), and test (20%) sets
    - Vectorize (tokenize) the raw text
    """

    # Load and pre-process the PAN 2018 English dataset and split 20% for test
    (processed_merged_tweets_trainval, processed_merged_tweets_test,
     labels_trainval, labels_test, author_ids_trainval, author_ids_test) = _load_split_pan18ap_training_corpus()

    # Vectorize the training+validation raw text and split it into stratified training and validation sets
    (x_train, x_val, y_train, y_val,
     tokenizer, word_index) = _vectorize_split_trainval(processed_merged_tweets_trainval, labels_trainval,
                                                        author_ids_trainval, MAX_WORDS, MAX_SEQUENCE_LEN)

    # Vectorize (tokenize) the test set raw text
    # Note that the tokenizer is already fit on the training+validation set
    sequences_test = tokenizer.texts_to_sequences(processed_merged_tweets_test)
    logger.info('@ %.2f seconds: Finished tokenizing the test set raw texts', time.process_time())
    #
    x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LEN)
    y_test = np.asarray(labels_test)

    logger.info('Shape of data  (x) tensor = {training: %s | validation: %s | test: %s}',
                x_train.shape, x_val.shape, x_test.shape)
    logger.info('Shape of label (y) tensor = {training: %s | validation: %s | test: %s}',
                y_train.shape, y_val.shape, y_test.shape)

    # TODO: GloVe: why did it fail?

    return x_train, x_val, x_test, y_train, y_val, y_test, word_index


def _load_split_pan18ap_training_corpus():
    """Load and pre-process the PAN 2018 English dataset and split 20% for test

    - Load the English training corpus of the PAN 2018 Author Profiling task
    - Pre-process the raw text (replace URLs, etc.)
    - Split 20% of the raw dataset for test set (stratified)
    """

    XMLS_DIRECTORY = 'data/PAN 2018, Author Profiling/en/text'
    TRUTH_PATH = 'data/PAN 2018, Author Profiling/en/en.txt'

    # Load the raw texts and the labels (truths) from the files into lists
    merged_tweets, text_labels, author_ids, original_tweet_lengths = \
        datasets.process_data_files.load_pan_data(XMLS_DIRECTORY, TRUTH_PATH)

    # Map textual labels to numeric labels:
    # 'female' → 0 and 'male' → 1
    labels = []  # Create an empty list
    for text_label in text_labels:
        if text_label == 'female':
            labels.append(0)
        elif text_label == 'male':
            labels.append(1)
        else:
            raise ValueError('The labels are expected to be "male" or "female". Encountered label "%s".' % text_label)

    # Process the merged tweets using NLTK's tweet tokenizer to replace repeated characters,
    # and remove URLs and @Username mentions
    processed_merged_tweets = []  # Create an empty list
    for merged_tweets_of_author in merged_tweets:
        processed_merged_tweets.append(preprocess_tweet(merged_tweets_of_author, replacement_tags=False))

    # Split the raw dataset into balanced (stratified) training+validation and test sets (split 20% for test set)
    processed_merged_tweets_trainval, processed_merged_tweets_test, labels_trainval, labels_test, \
    author_ids_trainval, author_ids_test = train_test_split(processed_merged_tweets, labels, author_ids,
                                                            test_size=0.2, random_state=42, stratify=labels)
    # ↳ *stratify=labels* selects a balanced sample from the data, with the same class proportion as the *labels* list.

    return (processed_merged_tweets_trainval, processed_merged_tweets_test,
            labels_trainval, labels_test, author_ids_trainval, author_ids_test)


def _load_pan18ap_test_corpus():
    """Load and pre-process the official test corpus of PAN 2018 (only English)

    - Load the English test corpus of the PAN 2018 Author Profiling task
    - Pre-process the raw text (replace URLs, etc.)

    This is used as a second test set for age classification experiments on the ASI dataset.
    """

    XMLS_DIRECTORY = 'data/PAN 2018, Author Profiling - Test Corpus/en/text'
    TRUTH_PATH = 'data/PAN 2018, Author Profiling - Test Corpus/en/en.txt'

    # Load the raw texts and the labels (truths) from the files into lists
    merged_tweets, text_labels, author_ids, _ignore = \
        datasets.process_data_files.load_pan_data(XMLS_DIRECTORY, TRUTH_PATH)

    # Map textual labels to numeric labels:
    # 'female' → 0 and 'male' → 1
    labels = []  # Create an empty list
    for text_label in text_labels:
        if text_label == 'female':
            labels.append(0)
        elif text_label == 'male':
            labels.append(1)
        else:
            raise ValueError('The labels are expected to be "male" or "female". Encountered label "%s".' % text_label)

    # Process the merged tweets using NLTK's tweet tokenizer to replace repeated characters,
    # and remove URLs and @Username mentions
    processed_merged_tweets = []  # Create an empty list
    for merged_tweets_of_author in merged_tweets:
        processed_merged_tweets.append(preprocess_tweet(merged_tweets_of_author, replacement_tags=False))

    return processed_merged_tweets, labels, author_ids


def _vectorize_split_trainval(processed_merged_tweets_trainval, labels_trainval, user_ids_trainval,
                              MAX_WORDS, MAX_SEQUENCE_LEN):
    """Vectorize the training+validation raw text and split it into stratified training and validation sets

    - Vectorize (tokenize) the training+validation raw text
    - Split the training+validation dataset into balanced (stratified) training and validation sets
    """

    logger.info("MAX_SEQUENCE_LEN = %s  |  MAX_WORDS = %s",
                format(MAX_SEQUENCE_LEN, ',d') if MAX_SEQUENCE_LEN is not None else None,
                format(MAX_WORDS, ',d') if MAX_WORDS is not None else None
                )
    # ↳ Work around the TypeError when one of the values is *None*.
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    # Build the word index
    tokenizer.fit_on_texts(processed_merged_tweets_trainval)
    # How you can recover the word index that was computed
    word_index = tokenizer.word_index
    # ↳ The word_index dictionary includes all the words in the documents.
    # Turn the strings into lists of integer indices.
    # Any word other than the *MAX_WORDS* most frequent words will be ignored in this process.
    sequences_trainval = tokenizer.texts_to_sequences(processed_merged_tweets_trainval)
    logger.info('@ %.2f seconds: Finished tokenizing the training+validation raw texts', time.process_time())
    logger.info('Found %s unique tokens.' % format(len(word_index), ',d'))
    #
    # Turn the list of integers into a 2D integer tensor of shape (samples, maxlen)
    x_trainval = pad_sequences(sequences_trainval, maxlen=MAX_SEQUENCE_LEN)
    #
    y_trainval = np.asarray(labels_trainval)

    # Split the training+validation dataset into balanced (stratified) training and validation sets
    # Note: 20% (validation set) is 25% of 80% (training+validation), hence the *test_size=0.25* option.
    x_train, x_val, y_train, y_val, user_ids_train, user_ids_val = \
        train_test_split(x_trainval, y_trainval, user_ids_trainval,
                         test_size=0.25, random_state=42, stratify=y_trainval)
    # ↳ Note: The array-like object given to the *stratify* option should have the same number of samples as the inputs.

    return x_train, x_val, y_train, y_val, tokenizer, word_index


def load_split_and_vectorize_asi_data(MAX_WORDS, MAX_SEQUENCE_LEN):
    """Load, vectorize and split the ASI (Advanced Symbolics Inc.) data

    - Load the preprocessed dataset (URLs and username mentions removed, repeated characters normalized,
        lowercased, etc.)
    - Split the dataset into balanced (stratified) training (60%), validation (20%), and test (20%) sets
    - Vectorize (tokenize) the raw text

    Also, load and vectorize the official test set of the PAN 2018 Author Profiling dataset as a second test set.
    """

    LABELS_XML_PATH = 'data/Advanced Symbolics/Labels.xml'
    TWEETS_XMLS_DIR = 'data/Advanced Symbolics/Tweets'

    # Load the raw texts and the labels from the files into lists
    user_ids, processed_merged_tweets, text_genders, text_ages =\
        datasets.asi.load(LABELS_XML_PATH, TWEETS_XMLS_DIR, stratified_subset='genders')

    # Map textual gender labels to numeric labels:
    # 'female' → 0 and 'male' → 1
    gender_labels = []
    for text_label in text_genders:
        if text_label == 'female':
            gender_labels.append(0)
        elif text_label == 'male':
            gender_labels.append(1)
        else:
            raise ValueError('The labels are expected to be "male" or "female". Encountered label "%s".' % text_label)

    # TODO (not for now): Map textual age labels to numeric labels

    # Split the raw dataset into balanced (stratified) training+validation and test sets (split 20% for test set)
    (processed_merged_tweets_trainval, processed_merged_tweets_test,
     gender_labels_trainval, gender_labels_test,
     user_ids_trainval, user_ids_test) = train_test_split(processed_merged_tweets, gender_labels, user_ids,
                                                          test_size=0.2, random_state=42, stratify=gender_labels)
    # ↳ *stratify=gender_labels* selects a balanced sample from the data, with the same class proportion as
    #   the *gender_labels* list.

    # Vectorize the training+validation raw text and split it into stratified training and validation sets
    (x_train, x_val, y_train, y_val,
     tokenizer, word_index) = _vectorize_split_trainval(processed_merged_tweets_trainval, gender_labels_trainval,
                                                        user_ids_trainval, MAX_WORDS, MAX_SEQUENCE_LEN)

    # Vectorize (tokenize) the ASI test set raw text
    # Note that the tokenizer is already fit on the training+validation set
    sequences_test_asi = tokenizer.texts_to_sequences(processed_merged_tweets_test)
    logger.info('@ %.2f seconds: Finished tokenizing the ASI test set raw texts', time.process_time())
    #
    x_test_asi = pad_sequences(sequences_test_asi, maxlen=MAX_SEQUENCE_LEN)
    y_test_asi = np.asarray(gender_labels_test)

    # Load and preprocess PAN 2018's official test corpus (only English) to use as a second test set for
    # the gender experiments
    pan18ap_processed_merged_tweets_test, pan18ap_labels_test, _ignored1 = _load_pan18ap_test_corpus()

    # Vectorize (tokenize) the PAN18AP test set raw text
    # Note that the tokenizer is already fit on the training+validation set
    sequences_test_pan18ap = tokenizer.texts_to_sequences(pan18ap_processed_merged_tweets_test)
    logger.info('@ %.2f seconds: Finished tokenizing the PAN18AP test set (official test corpus) raw texts',
                time.process_time())
    #
    x_test_pan18ap = pad_sequences(sequences_test_pan18ap, maxlen=MAX_SEQUENCE_LEN)
    y_test_pan18ap = np.asarray(pan18ap_labels_test)

    logger.info('Shape of data  (x) tensor = {training: %s | validation: %s | test_asi: %s | test_pan18ap: %s}',
                x_train.shape, x_val.shape, x_test_asi.shape, x_test_pan18ap.shape)
    logger.info('Shape of label (y) tensor = {training: %s | validation: %s | test_asi: %s | test_pan18ap: %s}',
                y_train.shape, y_val.shape, y_test_asi.shape, y_test_pan18ap.shape)

    return x_train, x_val, x_test_asi, x_test_pan18ap, y_train, y_val, y_test_asi, y_test_pan18ap, word_index


def prepare_glove_embeddings(MAX_WORDS, EMBEDDING_DIM, word_index):
    """Prepare the GloVe embeddings matrix

    Args:
        MAX_WORDS:     Size of the vocabulary. Consider only the *MAX_WORDS* most frequent words
        EMBEDDING_DIM: Word embeddings dimension
        word_index:    A dictionary mapping words to indexes in a Tokenizer object. The words in the dictionary are
        sorted by frequency of appearance in the dataset. Only the first *MAX_WORDS* items of the dictionary will be
        used here.

    Returns:
        The GloVe embedding matrix with shape (MAX_WORDS, EMBEDDING_DIM).
        This matrix can then be loaded into the Embedding layer of a model to set its weights.
    """

    # Load GloVe
    GLOVE_DIR = 'data/GloVe - Twitter Word Embeddings'
    GLOVE_PATH = os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt')
    embeddings_index = datasets.process_data_files.load_glove_embeddings(GLOVE_PATH)

    # Prepare the GloVe word embeddings matrix
    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))  # Initialize with zeros
    #
    for word, i in word_index.items():
        if i < MAX_WORDS:
            # Any word other than the *MAX_WORDS* most frequent words will be ignored.
            # Note that *word_index* includes all the tokens in the documents, not just the top *MAX_WORDS*.
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                # Words not found in the embedding index will be all zeros.

    logger.info('@ %.2f seconds: Finished preparing the GloVe word-embeddings matrix', time.process_time())

    return embedding_matrix


def serialize_model_and_history(model, history):
    """ Serialize (save) the trained model and its performance over time during training and validation

    - Log the SHA1 hash of the trained model and the history object (as an experiment reproducibility check)
    - Save the architecture of the model to a YAML file
    - Save the weights of the model to an HDF5 file
    - Pickle *history.history*—the record of training and validation loss and metrics at successive epochs

    Args:
        model: A trained model
        history: The history object returned by the *model.fit* method
    """

    logger.info('Experiment reproducibility check (SHA1 hash):')
    logger.info('    trained model: %s', my_utils.hex_hash_object(model))
    logger.info('    history:       %s', my_utils.hex_hash_object(history))

    # • Serialize (save) the trained model
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    MODELS_DIR = 'data/out/models'
    YAML_FILENAME = RUN_TIMESTAMP + ' ' + 'model architecture' + '.yaml'
    WEIGHTS_FILENAME = RUN_TIMESTAMP + ' ' + 'model weights' + '.h5'
    # Create the directory if it does not exist.
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save the architecture of the model (not its weights or its training configuration) to a YAML file
    # YAML (compared to JSON) is more suitable for configuration files.
    model_as_yaml_string = model.to_yaml()
    with open(os.path.join(MODELS_DIR, YAML_FILENAME), 'w') as yaml_file:
        yaml_file.write(model_as_yaml_string)
    # Save the weights of the model to an HDF5 file
    model.save_weights(os.path.join(MODELS_DIR, WEIGHTS_FILENAME))

    # • Pickle *history.history*
    PICKLES_DIR = 'data/out/pickles'
    HISTORY_PICKLE_FILENAME = RUN_TIMESTAMP + ' ' + 'history' + '.pickle'
    # Create the directory if it does not exist.
    os.makedirs(PICKLES_DIR, exist_ok=True)
    # Pickle
    with open(os.path.join(PICKLES_DIR, HISTORY_PICKLE_FILENAME), 'wb') as pickle_output_file:
        pickle.dump(history.history, pickle_output_file)


def log_plot_training_performance(history, PLOT=True):
    """Plot and log the model's performance over time during training and validation

    Args:
        history: The history object returned by the *model.fit* method in Keras.
        PLOT: If True (default), will plot the model's performance. If False, will only log the model's performance.
    """

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    logger.info('Training accuracy: %s', acc)
    logger.info('Validation accuracy: %s', val_acc)
    logger.info('Training loss: %s', loss)
    logger.info('Validation loss: %s', val_loss)

    if PLOT:
        # Create a figure with two subplots
        figure, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(epochs, acc, '.:', label='Training accuracy')
        ax1.plot(epochs, val_acc, 'o-', label='Validation accuracy')
        ax1.set_ylabel('accuracy')
        ax1.legend()

        ax2.plot(epochs, loss, '.:', label='Training loss')
        ax2.plot(epochs, val_loss, 'o-', label='Validation loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend()

        figure.suptitle('Accuracy and loss for the training and validation')

        plt.show()


def evaluate_model_on_test_set(model, x_test, y_test, TEST_SET_LABEL=''):
    """Evaluate the model (already trained) on the test set"""

    # Add a space character before the test set label if it's not blank
    space_control = ' ' if TEST_SET_LABEL != '' else ''

    metrics_values = model.evaluate(x_test, y_test)
    logger.info('@ %.2f seconds: Finished evaluating the model on the%s%s test set',
                time.process_time(), space_control, TEST_SET_LABEL)
    for name, value in zip(model.metrics_names, metrics_values):
        logger.info("%s = %s", name, value)


def main():
    """The main function.

    Every time the script runs, it will call this function.
    """

    logger.info('Experiment notes: --')

    # Ensure reproducible results
    ensure_reproducibility()

    # Size of the vocabulary—Consider only the 10,000 most frequent words in the dataset as features
    MAX_WORDS = 10 ** 4
    # Length of sequences—Cut off the text after this many words (among the *MAX_WORDS* most common words)
    # Can be passed as *None* to the tokenizer, but not to the embedding layer when you are going to connect
    # Flatten and Dense layers upstream. More info: https://keras.io/layers/embeddings/
    MAX_SEQUENCE_LEN = 2644  # 20 * 1000

    # (x_train, x_val, x_test,
    #  y_train, y_val, y_test, word_index) = load_split_and_vectorize_pan18ap_data(MAX_WORDS, MAX_SEQUENCE_LEN)

    (x_train, x_val, x_test_asi, x_test_pan18ap,
     y_train, y_val, y_test_asi, y_test_pan18ap,
     word_index) = load_split_and_vectorize_asi_data(MAX_WORDS, MAX_SEQUENCE_LEN)

    trained_model, history = def_train_model.fully_connected_with_dropout_l2(
        x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index)
    # trained_model, history = def_train_model.fully_connected_with_dropout_l2(
    #     x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index)

    serialize_model_and_history(trained_model, history)
    log_plot_training_performance(history, PLOT=False)  # Note: Cannot plot on Compute Canada nodes.

    # evaluate_model_on_test_set(trained_model, x_test, y_test)

    evaluate_model_on_test_set(trained_model, x_test_asi, y_test_asi, TEST_SET_LABEL='ASI')
    evaluate_model_on_test_set(trained_model, x_test_pan18ap, y_test_pan18ap, TEST_SET_LABEL='PAN18AP')

    # Destroy the current TF graph and create a new one, to ensure reproducible results.
    # This is also useful to avoid clutter from old models/layers.
    K.clear_session()

    # Log run time
    logger.info("@ %.2f seconds: Run finished", time.process_time())


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to '__main__' when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == '__main__':
    logger, RUN_TIMESTAMP = my_utils.configure_root_logger(1)
    my_utils.set_working_directory(1)
    main()
