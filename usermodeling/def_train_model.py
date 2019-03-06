"""Define and train deep learning models"""

import logging
import time

from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential


def basic_word_embeddings(x_train, x_val, y_train, y_val, MAX_WORDS, MAX_SEQUENCE_LEN, word_index):
    """Define and train a basic word embeddings model"""

    EMBEDDING_DIM = 100

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


'''
The following lines will be executed any time this .py file is run as a script or imported as a module.
'''
# Create a logger object. The root logger would be the parent of this logger
# Note that if you run this .py file as a script, this logger will not function, because it is not configured.
logger = logging.getLogger(__name__)
