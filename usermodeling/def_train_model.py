"""Define and train deep learning models

Each function in this module gets the vectorized dataset, defines a deep learning model, and trains it on the dataset.
"""

import logging
import time

from keras.layers import Embedding, Flatten, Dense, LSTM, GRU, Bidirectional, Dropout
from keras.models import Sequential
from keras import regularizers


def basic_fully_connected(x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index):
    """Define and train a basic, fully connected model with word embeddings"""

    EMBEDDING_DIM = 8

    # • Define the model
    model = Sequential()
    # Embedding layer
    '''
    After the Embedding layer, the  activations have shape (samples, MAX_SEQUENCE_LEN, EMBEDDING_DIM)
    
    Arguments:
        input_dim = MAX_WORDS:           Size of the vocabulary
        output_dim = EMBEDDING_DIM:      Dimension of the dense embedding
        input_length = MAX_SEQUENCE_LEN: Length of input sequences, when it is constant
    
    Shape of input and output:
        Input:    2D tensor with shape (batch_size, input_length)
        Output:   3D tensor with shape (batch_size, input_length, output_dim)
    '''
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN))
    # Flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, MAX_SEQUENCE_LEN * EMBEDDING_DIM)
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    # Add the classifier on top
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=logger.info)

    # # Load the pre-trained word embeddings (GloVe) into the Embedding layer
    # glove_embedding_matrix = prepare_glove_embeddings(MAX_WORDS, EMBEDDING_DIM, word_index)
    # model.layers[0].set_weights([glove_embedding_matrix])
    # # Freeze the Embedding layer
    # model.layers[0].trainable = False

    # Compile and train the model (and evaluate it on the validation set)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        )

    logger.info('@ %.2f seconds: Finished training and validation', time.process_time())

    return model, history


def fully_connected_with_dropout_l2(x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index):
    """Define and train a fully connected model with word embeddings, dropout, and L2 weight regularization"""

    EMBEDDING_DIM = 8

    # • Define the model
    model = Sequential()
    # Embedding layer
    '''
    After the Embedding layer, the  activations have shape (samples, MAX_SEQUENCE_LEN, EMBEDDING_DIM)

    Arguments:
        input_dim = MAX_WORDS:           Size of the vocabulary
        output_dim = EMBEDDING_DIM:      Dimension of the dense embedding
        input_length = MAX_SEQUENCE_LEN: Length of input sequences, when it is constant

    Shape of input and output:
        Input:    2D tensor with shape (batch_size, input_length)
        Output:   3D tensor with shape (batch_size, input_length, output_dim)
    '''
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN))
    # Flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, MAX_SEQUENCE_LEN * EMBEDDING_DIM)
    model.add(Flatten())
    # Dropout layer 1
    model.add(Dropout(0.5))
    #
    model.add(Dense(32,
                    kernel_regularizer=regularizers.l2(10 ** -2),
                    activation='relu'))
    # Dropout layer 2
    model.add(Dropout(0.5))
    # Add the classifier on top
    model.add(Dense(1, activation='sigmoid'))

    model.summary(print_fn=logger.info)

    # # Load the pre-trained word embeddings (GloVe) into the Embedding layer
    # glove_embedding_matrix = prepare_glove_embeddings(MAX_WORDS, EMBEDDING_DIM, word_index)
    # model.layers[0].set_weights([glove_embedding_matrix])
    # # Freeze the Embedding layer
    # model.layers[0].trainable = False

    # Compile and train the model (and evaluate it on the validation set)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    history = model.fit(x_train, y_train,
                        epochs=40,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        )

    logger.info('@ %.2f seconds: Finished training and validation', time.process_time())

    return model, history


def rnn(x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index):
    """Define and train an RNN (LSTM/GRU) model"""

    EMBEDDING_DIM = 32

    # • Define the model
    model = Sequential()
    # Embedding layer
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM))
    # model.add(LSTM(32))
    model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))  # Add dropout to fight overfitting
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=logger.info)

    # Compile and train the model (and evaluate it on the validation set)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=(x_val, y_val),
                        )

    logger.info('@ %.2f seconds: Finished training and validation', time.process_time())

    return model, history


def bidirectional_rnn(x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index):
    """Define and train a bidirectional RNN (LSTM/GRU) model"""

    EMBEDDING_DIM = 32

    # • Define the model
    model = Sequential()
    # Embedding layer
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=logger.info)

    # Compile and train the model (and evaluate it on the validation set)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_data=(x_val, y_val),
                        )

    logger.info('@ %.2f seconds: Finished training and validation', time.process_time())

    return model, history


def bidirectional_rnn_with_dropout(x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index):
    """Define and train a bidirectional RNN (LSTM/GRU) model with dropout"""

    EMBEDDING_DIM = 32

    # • Define the model
    model = Sequential()
    # Embedding layer
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM))
    model.add(Bidirectional(LSTM(32,
                                 dropout=0.2,
                                 recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=logger.info)

    # Compile and train the model (and evaluate it on the validation set)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    history = model.fit(x_train, y_train,
                        epochs=40,
                        batch_size=128,
                        validation_data=(x_val, y_val),
                        )

    logger.info('@ %.2f seconds: Finished training and validation', time.process_time())

    return model, history


'''
The following lines will be executed any time this .py file is run as a script or imported as a module.
'''
# Create a logger object. The root logger would be the parent of this logger
# Note that if you run this .py file as a script, this logger will not function, because it is not configured.
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # The following lines will be executed only if this .py file is run as a script,
    # and not if it is imported as a module.
    print("Module was executed directly.")
