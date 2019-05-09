"""Hypothesis testing and compare experiment results"""

from io import BytesIO
import logging
import os
import pickle

from matplotlib import pyplot as plt
from pandas import DataFrame
import pyperclip as clipboard
from scipy.stats import normaltest
from scipy.stats import ttest_ind

from usermodeling.utils import my_utils

# Change the level of the loggers of some of the imported modules
logging.getLogger("matplotlib").setLevel(logging.INFO)


def legacy_func():
    """Old function used for PAN18"""

    results = DataFrame()

    # experiment_names = ('N-grams', 'N-grams, LSA 300')
    # results[experiment_names[0]] = [0.8, 0.82222222, 0.80555556, 0.82222222, 0.82222222, 0.79444444, 0.87777778, 0.85, 0.81666667, 0.77777778]
    # results[experiment_names[1]] = [0.82777778, 0.84444444, 0.84444444, 0.83888889, 0.81111111, 0.8, 0.88888889, 0.82777778, 0.78333333, 0.81666667]
    #
    # # %%% TEMP: Testing
    # temp_list = []
    # for item in results[experiment_names[0]]:
    #     temp_list.append(item - 0.02)
    # results[experiment_names[0]] = temp_list
    #
    experiment_names = ('Word and char n-grams, No LSA (English)', 'Word n-grams, No LSA (English)')
    results[experiment_names[0]] = [0.8, 0.79444444, 0.82222222, 0.87222222, 0.76111111, 0.85, 0.82222222, 0.85555556, 0.81111111, 0.81111111]
    results[experiment_names[1]] = [0.8, 0.78333333, 0.81111111, 0.85, 0.77222222, 0.81666667, 0.80555556, 0.82777778, 0.8, 0.8]


    # experiment_names = ('Word and char n-grams, No LSA (Arabic)', 'Char n-grams, No LSA (Arabic)')
    # results[experiment_names[0]] = [0.83333333, 0.77777778, 0.81111111, 0.76666667, 0.81111111, 0.87777778, 0.81111111, 0.77777778, 0.81111111, 0.75555556]
    # results[experiment_names[1]] = [0.77777778, 0.75555556, 0.83333333, 0.74444444, 0.78888889, 0.82222222, 0.82222222, 0.75555556, 0.77777778, 0.71111111]

    # Descriptive stats
    print(results.describe())

    # Box and whisker plot
    results.boxplot()
    plt.show()

    # Histogram plot
    results.hist()
    plt.show()

    # Normality test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    alpha = 0.05
    print("Confidence level = {}%".format((1-alpha)*100))
    for experiment_name in experiment_names:
        statistic, p = normaltest(results[experiment_name])
        # ↳ Null hypothesis: the sample comes from a normal distribution
        print("{}: (skewtest z-score)^2 + (kurtosistest z-score)^2 = {}, p = {}".format(experiment_name, statistic, p))
        if p < alpha:
            # The null hypothesis can be rejected
            print("No: It is unlikely that the sample comes from a Gaussian (normal) distribution")
        else:
            print("Yes: It is likely that the sample comes from a Gaussian (normal) distribution")

    # T-test: Compare means for Gaussian samples
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    alpha = 0.05
    print("Confidence level = {}%".format((1-alpha)*100))
    equal_variances = True
    statistic, p = ttest_ind(results[experiment_names[0]], results[experiment_names[1]], equal_var=equal_variances)
    # ↳ Null hypothesis: The two samples have identical average (expected) values.
    if equal_variances:
        statistic_name = "t-test statistic"
    else:
        statistic_name = "Welch’s t-test statistic"

    print("{} = {}, p = {}".format(statistic_name, statistic, p))
    if p < alpha:
        # The null hypothesis can be rejected
        print("Significant difference between the means: Samples are likely drawn from different distributions")
    else:
        print("No significant difference between the means: Samples are likely drawn from the same distribution")


def unpickle_history(run_timestamp_and_title, mode=None):
    """Load a pickled *history.history* dictionary and return its information as lists

    - This function gets a timestamp string as input and loads its corresponding pickled *history.history* dictionary.
    - The history object is returned by the *model.fit* method in Keras. The *history.history* dictionary is an
    attribute of history which contains the performance of a model over time during training and validation.
    - This is useful to review and compare the performance of the models in previous experiments.

    Args:
        - run_timestamp_and_title: A string containing the date and time of the target run with the format
        '%Y-%m-%d_%H-%M-%S'. This string can also contain other characters after the timestamp. As long as the string
        begins with a timestamp in the above format, and there is some whitespace characters between the timestamp
        and the rest of the string, the rest of the string will be trimmed and ignored.
        - mode: If 'compute canada', some adjustments will be made to the pickles directory and the pickled file before
        unpickling, as a workaround to a bug in NumPy 1.16.0.

    Returns:
        epochs <range>: Epoch numbers
        acc <list of float64>: Training accuracies
        val_acc <list of float64>: Validation accuracies
        loss <list of float64>: Training losses
        val_loss <list of float64>: Validation losses
    """

    # In case the input character contains other characters after the timestring, trim the rest of it
    run_timestamp = run_timestamp_and_title.split()[0]

    if mode == 'compute canada':
        PICKLES_DIR = 'data/out/. Compute Canada/pickles'
    else:
        PICKLES_DIR = 'data/out/pickles'

    HISTORY_PICKLE_FILENAME = run_timestamp + ' ' + 'history' + '.pickle'

    # Unpickle the *history.history* dictionary
    with open(os.path.join(PICKLES_DIR, HISTORY_PICKLE_FILENAME), 'rb') as pickle_input_file:
        if mode == 'compute canada':
            # Quick workaround: Replace the "cnumpy.core._multiarray_umath" string with "cnumpy.core.multiarray"
            # Due to bug in NumPy 1.16.0, on the first line of the pickles written by Compute Canada nodes,
            # the above string is different. More info: https://github.com/numpy/numpy/issues/12977
            pickle_as_bytestring = pickle_input_file.read()
            pickle_as_bytestring = pickle_as_bytestring.replace(b'._multiarray_umath', b'.multiarray')
            new_pickle_file = BytesIO(pickle_as_bytestring)
            history_dot_history = pickle.load(new_pickle_file)
        else:
            # The regular unpickling
            history_dot_history = pickle.load(pickle_input_file)

    acc = history_dot_history['acc']
    val_acc = history_dot_history['val_acc']
    loss = history_dot_history['loss']
    val_loss = history_dot_history['val_loss']

    epochs = range(1, len(acc) + 1)

    return epochs, acc, val_acc, loss, val_loss


def plot_training_performance_from_pickle(run_timestamp_and_title, mode=None):
    """Load a pickled *history.history* dictionary and plot its information

    For information about the input arguments, refer to the docstring of the *unpickle_history()* function.
    """

    epochs, acc, val_acc, loss, val_loss = unpickle_history(run_timestamp_and_title, mode=mode)

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

    # Set limits of the y axis for both subplots
    y1 = (0, 1)
    y2 = (0, 3)
    #
    y1_margin = 0.05 * (y1[1] - y1[0])
    y2_margin = 0.05 * (y2[1] - y2[0])
    ax1.set_ylim(y1[0] - y1_margin, y1[1] + y1_margin)
    ax2.set_ylim(y2[0] - y2_margin, y2[1] + y2_margin)

    # Set limits of the x axis for both subplots
    x = (1, 10)
    #
    x_margin = 0.05 * (x[1] - x[0])
    xx = (x[0] - x_margin, x[1] + x_margin)
    ax1.set_xlim(xx)
    ax2.set_xlim(xx)

    figure.suptitle(run_timestamp_and_title + '\n' + 'Accuracy and loss for the training and validation')


def history_to_tabular_string(run_timestamp_and_title, mode=None):
    """Load a pickled *history.history* dictionary and return its information as a tabular text string

    - The output text string is copied onto the system clipboard, and is in the following form:
        Epoch	Training accuracy   Validation accuracy     Training loss   Validation loss
        1       1.0                 1.0                     1.0             1.0
        2       ...
        ...
        10      ...
        *run_timestamp_and_title*
    - The string is tabular, except the last line which is the run timestamp and title (the input of the function).
    - For information about the input arguments, refer to the docstring of the *unpickle_history()* function.
    """

    epochs, acc, val_acc, loss, val_loss = unpickle_history(run_timestamp_and_title, mode=mode)

    header = ['Epoch', 'Training accuracy', 'Validation accuracy', 'Training loss', 'Validation loss']
    rows = list(zip(epochs, acc, val_acc, loss, val_loss))

    # Turn the header into a tabular string and add it to *lines* list
    lines = ['\t'.join(header)]
    # Turn all the other rows into tabular strings and add them to the *lines* list
    lines.extend('\t'.join([str(item) for item in row]) for row in rows)
    # Add the run timestamp and title to the *lines* list (to help match the output to the run)
    lines.append(run_timestamp_and_title)
    # Join all the lines into a single string
    tabular_string = '\n'.join(lines)

    # Copy the string to the clipboard (using the *pyperclip* package)
    clipboard.copy(tabular_string)

    logger.info('The tabular string was copied to the clipboard.')


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to '__main__' when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == '__main__':
    logger = my_utils.configure_basic_logger()
    my_utils.set_working_directory(1)
    # legacy_func()

    # plot_training_performance_from_pickle('2019-04-12_18-44-51 deep_learning __ ASI, basic fully connected model, maxlen=20k', mode='compute canada')
    # plot_training_performance_from_pickle('2019-04-12_18-07-33 deep_learning __ ASI, basic fully connected model, maxlen=2,644', mode='compute canada')

    # plot_training_performance_from_pickle('2019-04-12_18-06-59 deep_learning __ ASI, basic fully connected model')
    # plot_training_performance_from_pickle('2019-04-25_20-48-32 deep_learning __ ASI, fully connected + dropout')
    # plot_training_performance_from_pickle('2019-04-25_21-20-31 deep_learning __ ASI, fully connected + 2 dropouts')

    # plot_training_performance_from_pickle('2019-04-25_21-20-31 deep_learning __ ASI, fully connected + 2 dropouts')
    # plot_training_performance_from_pickle('2019-04-25_21-49-49 deep_learning __ ASI, fully connected + dropout + L2 regularization')
    # plot_training_performance_from_pickle('2019-04-25_22-48-58 deep_learning __ ASI, fully connected + dropout + L2 regularization = 0.01')
    # plot_training_performance_from_pickle('2019-04-25_23-35-05 deep_learning __”, 40 epochs')
    # plot_training_performance_from_pickle('2019-03-30_00-59-55 deep_learning __ bidirectional LSTM, ASI')

    # plot_training_performance_from_pickle('2019-04-26_03-34-39 deep_learning __ ”, max_words=10^5')
    # plot_training_performance_from_pickle('2019-04-25_21-19-28 deep_learning __ ASI, fully connected + dropout + L2 regularization = 0.01, maxlen=20k', mode='compute canada')
    # plot_training_performance_from_pickle('2019-04-26_00-39-31 deep_learning __ ”, max_words=10^5', mode='compute canada')
    # plot_training_performance_from_pickle('2019-04-26_11-24-49 deep_learning __ max_words=10^4, maxlen=2,644. embedding_dim=100', mode='compute canada')

    # plot_training_performance_from_pickle('2019-04-09_21-22-52 deep_learning __ ASI, 22k users, maxlen=2,644', mode='compute canada')
    # plot_training_performance_from_pickle('2019-03-30_00-59-55 deep_learning __ bidirectional LSTM, ASI')
    # plot_training_performance_from_pickle('2019-04-26_16-03-39 deep_learning __ Bidirectional LSTM + dropout=0.2', mode='compute canada')
    # plot_training_performance_from_pickle('2019-04-28_22-27-52 deep_learning __ Bidirectional LSTM + dropout=0.1')
    #
    # plot_training_performance_from_pickle('2019-04-29_16-31-28 deep_learning __ 2 bidiractional LSTM layers + dropout=0.1')
    # plot_training_performance_from_pickle('2019-04-30_05-20-56 deep_learning __ [Duplicate] 2 bidiractional LSTM layers + dropout=0.1', mode='compute canada')

    # plot_training_performance_from_pickle('2019-05-01_21-35-41 deep_learning __ ”, + input & recurrent L2 regularization = 0.001', mode='compute canada')
    # plot_training_performance_from_pickle('2019-05-01_20-41-11 deep_learning __ ”, + input & recurrent L2 regularization = 0.01', mode='compute canada')
    # plt.show()

    history_to_tabular_string('2019-04-12_18-06-59 deep_learning __ ASI, basic fully connected model')
