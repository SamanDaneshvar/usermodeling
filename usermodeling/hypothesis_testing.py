"""Hypothesis testing and compare experiment results"""

import logging
import os
import pickle

from matplotlib import pyplot as plt
from pandas import DataFrame
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
    # experiment_names = ('Word and char n-grams, No LSA (English)', 'Word n-grams, No LSA (English)')
    # results[experiment_names[0]] = [0.8, 0.79444444, 0.82222222, 0.87222222, 0.76111111, 0.85, 0.82222222, 0.85555556, 0.81111111, 0.81111111]
    # results[experiment_names[1]] = [0.8, 0.78333333, 0.81111111, 0.85, 0.77222222, 0.81666667, 0.80555556, 0.82777778, 0.8, 0.8]


    experiment_names = ('Word and char n-grams, No LSA (Arabic)', 'Char n-grams, No LSA (Arabic)')
    results[experiment_names[0]] = [0.83333333, 0.77777778, 0.81111111, 0.76666667, 0.81111111, 0.87777778, 0.81111111, 0.77777778, 0.81111111, 0.75555556]
    results[experiment_names[1]] = [0.77777778, 0.75555556, 0.83333333, 0.74444444, 0.78888889, 0.82222222, 0.82222222, 0.75555556, 0.77777778, 0.71111111]

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


def plot_training_performance_from_pickle(run_timestamp):
    """Load a pickled *history.history* dictionary and plot its information

    This function gets a timestamp string as input and loads its corresponding pickled *history.history* dictionary.
    The history object is returned by the *model.fit* method in Keras. The *history.history* dictionary is an attribute
    of history which contains the performance of a model over time during training and validation.
    The function will plot this information.

    This is useful to review and compare the performance of the models in previous experiments.

    Args:
        run_timestamp: A string containing the date and time of the target run with the format '%Y-%m-%d_%H-%M-%S'.
        This string can also contain other characters after the timestamp. As long as the string begins with a
        timestamp in the above format, and there is some whitespace characters between the timestamp and the rest of
        the string, the rest of the string will be trimmed and ignored.
    """

    # In case the input character contains other characters after the timestring, trim the rest of it
    run_timestamp = run_timestamp.split()[0]

    PICKLES_DIR = 'data/out/pickles'
    HISTORY_PICKLE_FILENAME = run_timestamp + ' ' + 'history' + '.pickle'

    # Unpickle the *history.history* dictionary
    with open(os.path.join(PICKLES_DIR, HISTORY_PICKLE_FILENAME), 'rb') as pickle_input_file:
        history_dot_history = pickle.load(pickle_input_file)

    acc = history_dot_history['acc']
    val_acc = history_dot_history['val_acc']
    loss = history_dot_history['loss']
    val_loss = history_dot_history['val_loss']

    epochs = range(1, len(acc) + 1)

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

    figure.suptitle(run_timestamp + '\n' + 'Accuracy and loss for the training and validation')


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to "__main__" when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    logger = my_utils.configure_basic_logger()
    my_utils.set_working_directory()
    # legacy_func()

    plot_training_performance_from_pickle('2019-03-06_15-42-48 deep_learning __ LSTM')
    # plot_training_performance_from_pickle('2019-03-06_17-50-01 deep_learning __ GRU')
    plt.show()
