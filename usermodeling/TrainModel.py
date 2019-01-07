""" This script builds a gender classification model on the dataset of the Author Profiling task at the
    PAN 2018 shared task.
    A linear Support Vector classifier is trained on text features.
    %%

    Author: Saman Daneshvar
"""

import ProcessDataFiles

import logging
import os
from datetime import datetime
import time
import numpy
import sys
import argparse
import re
import pickle
import hashlib
import base64

import numpy as np
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from matplotlib import pyplot as plt

# Change the level of the loggers of some of the imported modules
logging.getLogger("matplotlib").setLevel(logging.INFO)


def configureRootLogger():
    """ This function creates a logger and sets its configurations.
    """

    # Create a RootLogger object
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    ''' ↳ The logger discards any logging calls with a level of severity lower than the level of the logger.
        Next, each handler decides to accept/discard the call based on its own level.
        By setting the level of the logger to NOTSET, we hand the power to handlers, and we don't filter out anything
        at the entrance. In effect, this is the same as setting the level to DEBUG (lowest level possible).
    '''

    # Create a console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # Make sure the *logs* folder is created inside the script directory, regardless of the current working directory
    scriptDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))
    LOGS_DIRECTORY = os.path.join(scriptDirectory, "logs")
    # Create the directory if it does not exist
    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
    # Define the log file name
    LOG_FILE_NAME = datetime.today().strftime("%Y-%m-%d_%H-%M-%S.log")
    logFilePath = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

    # Create a file handler
    fileHandler = logging.FileHandler(logFilePath, encoding="utf-8")
    fileHandler.setLevel(logging.DEBUG)

    # Create a formatter and set it to the handlers
    formatter = logging.Formatter("%(name)-16s: %(levelname)-8s %(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


def checkSystemInfo():
    """ This function logs current date and time, computer and user name, and script path.
        It also ensures that the current working directory is the same as the script directory.
    """

    # Log current date and time, computer and user name, and script path
    logger.info("Current date and time: %s", datetime.today())
    logger.info("Computer and user name: %s, %s", os.getenv('COMPUTERNAME'), os.getlogin())
    # ↳ For a full list of environment variables and their values, call *os.environ*
    scriptPath = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname or not)
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.
    logger.info("Script path: %s", scriptPath)

    # Check if the current working directory is the same as the script directory, and if not change it.
    scriptDirectory = os.path.dirname(scriptPath)
    if os.getcwd() == scriptDirectory:
        logger.info("Current working directory = Script directory"
                    "\n")
    else:
        logger.warning("Changing working directory from: %s", os.getcwd())
        # Change the working directory to the script directory
        os.chdir(scriptDirectory)
        logger.info("Current working directory: %s"
                    "\n", os.getcwd())


def loadDatasets_Development(presetKey):
    """ This function loads the PAN training dataset and truth by calling the *ProcessDataFiles* module,
        then splits the dataset into training and test sets.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'datasetName': 'PAN 2018 English',
                                            'xmlsDirectory': 'data/en/text',
                                            'truthPath': 'data/en/en.txt',
                                            'txtsDestinationDirectory': 'data/TXT Files/en',
                                            },
                          'PAN18_Spanish': {'datasetName': 'PAN 2018 Spanish',
                                            'xmlsDirectory': 'data/es/text',
                                            'truthPath': 'data/es/es.txt',
                                            'txtsDestinationDirectory': 'data/TXT Files/es',
                                            },
                          'PAN18_Arabic': {'datasetName': 'PAN 2018 Arabic',
                                            'xmlsDirectory': 'data/ar/text',
                                            'truthPath': 'data/ar/ar.txt',
                                            'txtsDestinationDirectory': 'data/TXT Files/ar',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[presetKey]

    # Load the PAN 2018 training dataset and the truth from the files into lists
    logger.info("Loading the %s training dataset and the truth...", PRESET['datasetName'])
    mergedTweetsOfAuthors, truths, authorIDs, originalTweetLengths =\
        ProcessDataFiles.loadPanData(PRESET['xmlsDirectory'], PRESET['truthPath'],
                                     False, PRESET['txtsDestinationDirectory'])

    # Split the dataset into balanced (stratified) training and test sets:
    docs_train, docs_test, y_train, y_test, authorIDs_train, authorIDs_test,\
    originalTweetLengths_train, originalTweetLengths_test =\
        train_test_split(mergedTweetsOfAuthors, truths, authorIDs, originalTweetLengths,
                         test_size=0.4, random_state=42, stratify=truths)
    # ↳ *stratify=truths* selects a balanced sample from the data, with the same class proportion as the *truths* list.

    # • Sort all lists in the ascending order of *authorIDs* (separately, for the training and test set)
    # This is only done for the sakes of consistency between the loadDatasets_Development() and
    # loadDatasets_TiraEvaluation() functions, because the output of the latter is sorted by *authorIDs*, while the
    # former is shuffled by the *train_test_split()* function.
    # Sort the training set
    authorIDs_train, docs_train, y_train, originalTweetLengths_train = [list(tuple) for tuple in zip(*sorted(zip(
        authorIDs_train, docs_train, y_train, originalTweetLengths_train)))]
    # Sort the test set
    authorIDs_test, docs_test, y_test, originalTweetLengths_test = [list(tuple) for tuple in zip(*sorted(zip(
        authorIDs_test, docs_test, y_test, originalTweetLengths_test)))]

    # # TEMP: Used for producing a mimic of the **TIRA** environment
    # ProcessDataFiles.splitTrainAndTestFiles(authorIDs_train, authorIDs_test, y_train, y_test, presetKey)

    return docs_train, docs_test, y_train, y_test


def loadDatasets_TiraEvaluation(testDatasetMainDirectory, presetKey):
    """ This function loads the PAN training and test dataset and truth by calling the *ProcessDataFiles* module twice,
        then passes them along with Author IDs of the test dataset.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'datasetName': 'PAN 2018 English',
                                            'xmlsSubdirectory': 'en/text',
                                            'truthSubpath': 'en/truth.txt',
                                            },
                          'PAN18_Spanish': {'datasetName': 'PAN 2018 Spanish',
                                            'xmlsSubdirectory': 'es/text',
                                            'truthSubpath': 'es/truth.txt',
                                            },
                          'PAN18_Arabic': {'datasetName': 'PAN 2018 Arabic',
                                            'xmlsSubdirectory': 'ar/text',
                                            'truthSubpath': 'ar/truth.txt',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[presetKey]

    # Define the constant and the paths
    TRAINING_DATASET_MAIN_DIRECTORY =\
        "//VBOXSVR/training-datasets/author-profiling/pan18-author-profiling-training-dataset-2018-02-27"

    # # TEMP (TIRA): For local testing on SaMaN-Laptop
    # TRAINING_DATASET_MAIN_DIRECTORY = "C:/Users/Saman/PycharmProjects/PAN18_AuthorProfiling/data/TiraDummy/trainDirectory"

    # # TEMP (TIRA): For local testing on TIRA
    # TRAINING_DATASET_MAIN_DIRECTORY = "E:/author-profiling/pan18-author-profiling-training-dataset-2018-02-27"

    xmlsDirectory_train = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['xmlsSubdirectory'])
    truthPath_train = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['truthSubpath'])
    xmlsDirectory_test = os.path.join(testDatasetMainDirectory, PRESET['xmlsSubdirectory'])
    # ↳ Note: truthPath_test will not be provided to the participants.

    # Load the PAN 2018 training dataset and truth from the files into lists
    logger.info("Loading the %s training dataset and truth...", PRESET['datasetName'])
    docs_train, y_train, authorIDs_train, originalTweetLengths_train = \
        ProcessDataFiles.loadPanData(xmlsDirectory_train, truthPath_train, False, None)

    # Load the PAN 2018 test dataset from the files into lists
    logger.info("Loading the %s test dataset...", PRESET['datasetName'])
    docs_test, y_test, authorIDs_test, originalTweetLengths_test = \
        ProcessDataFiles.loadPanData(xmlsDirectory_test, None, False, None)
    # ↳ Note: truthPath_test will not be provided to the participants. As a result, *truths_test* will be empty.

    return docs_train, docs_test, y_train, authorIDs_test


def preprocessTweet(tweet):
    """ This function gets a string as input and outputs a string, doing the following pre-processing operations:
        - Replaces repeated character sequences of length 3 or greater with sequences of length 3
        - Lowercases
        - List of replacements: %%
            URL		    <URLURL>
            @Username   <UsernameMention>

        BETA VERSION:
        - Instead of tokenizing and detokenizing, which is messy, I should directly replace the strings using regex.
    """

    replacedURLs = []  # Create an empty list
    replacedMentions = []  # Create an empty list

    # Tokenize using NLTK
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)

    # Iterate over tokens
    for index, token in enumerate(tokens):
        # Replace URLs
        if token[0:8] == "https://":
            replacedURLs.append(token)
            tokens[index] = "<URLURL>"
            # ↳ *tokens[index]* will directly modify *tokens*, whereas any changes to *token* will be lost.

        # Replace mentions (Twitter handles; usernames)
        elif token[0] == "@" and len(token) > 1:
            # ↳ Skip the single '@' tokens
            replacedMentions.append(token)
            tokens[index] = "<UsernameMention>"

    # Detokenize using NLTK's Treebank Word Detokenizer
    detokenizer = TreebankWordDetokenizer()
    processedTweet = detokenizer.detokenize(tokens)

    # *replacedURLs* and *replacedMentions* will contain all of the replaced URLs and Mentions of the input string.
    return processedTweet


def extractFeatures(docs_train, docs_test, presetKey):
    """ This function builds a transformer (vectorizer) pipeline,
        fits the transformer to the training set (learns vocabulary and idf),
        transforms the training set and the test set to their TF-IDF matrix representation,
        and builds a classifier.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'datasetName': 'PAN 2018 English',
                                            'word_ngram_range': (1, 3),
                                            'performDimentionalityReduction': False, # %%%% TEMP: No LSA!
                                            },
                          'PAN18_Spanish': {'datasetName': 'PAN 2018 Spanish',
                                            'word_ngram_range': (1, 2),
                                            'performDimentionalityReduction': False,
                                            },
                          'PAN18_Arabic': {'datasetName': 'PAN 2018 Arabic',
                                            'word_ngram_range': (1, 2),
                                            'performDimentionalityReduction': False,
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[presetKey]

    # Build a vectorizer that splits strings into sequences of i to j words
    wordVectorizer = TfidfVectorizer(preprocessor=preprocessTweet,
                                     analyzer='word', ngram_range=PRESET['word_ngram_range'],
                                     min_df=2, use_idf=True, sublinear_tf=True)
    # Build a vectorizer that splits strings into sequences of 3 to 5 characters
    charVectorizer = TfidfVectorizer(preprocessor=preprocessTweet,
                                     analyzer='char', ngram_range=(3, 5),
                                     min_df=2, use_idf=True, sublinear_tf=True)
    # %% Trying out count vectorizer
    # vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word', min_df=1)

    # Build a transformer (vectorizer) pipeline using the previous analyzers
    # FeatureUnion concatenates results of multiple transformer objects
    ngramsVectorizer = Pipeline([('feats', FeatureUnion([('word_ngram', wordVectorizer),
                                                         ('char_ngram', charVectorizer),
                                                         ])),
                                 # ('clff', LinearSVC(random_state=42))
                                 ])

    # Fit (learn vocabulary and IDF) and transform (transform documents to the TF-IDF matrix) the training set
    X_train_ngramsTfidf = ngramsVectorizer.fit_transform(docs_train)
    ''' ↳ Check the following attributes of each of the transformers (analyzers)—*wordVectorizer* and *charVectorizer*:
        vocabulary_ : dict. A mapping of terms to feature indices.
        stop_words_ : set. Terms that were ignored
    '''
    logger.info("@ %.2f seconds: Finished fit_transforming the training dataset", time.process_time())
    logger.info("Training set word & character ngrams .shape = %s", X_train_ngramsTfidf.shape)

    # %%% Feature names %%%% TEMP: no featureNames!
    featureNames_ngrams = []
    # featureNames_ngrams = [wordVectorizer.vocabulary_, charVectorizer.vocabulary_]

    # # TEMP: For debugging purposes
    # ProcessDataFiles.writeIterableToCSV(list(featureNames_ngrams[0].items()), "wordVectorizer.vocabulary_",
    #                                     logger.handlers[1].baseFilename)
    # ProcessDataFiles.writeIterableToCSV(list(featureNames_ngrams[1].items()), "charVectorizer.vocabulary_",
    #                                     logger.handlers[1].baseFilename)

    ''' Extract the features of the test set (transform test documents to the TF-IDF matrix)
    Only transform is called on the transformer (vectorizer), because it has already been fit to the training set.
    '''
    X_test_ngramsTfidf = ngramsVectorizer.transform(docs_test)
    logger.info("@ %.2f seconds: Finished transforming the test dataset", time.process_time())
    logger.info("Test set word & character ngrams .shape = %s", X_test_ngramsTfidf.shape)

    # • Dimensionality reduction using truncated SVD (aka LSA)
    if PRESET['performDimentionalityReduction']:
        # Build a truncated SVD (LSA) transformer object
        svd = TruncatedSVD(n_components=300, random_state=43) # %%%
        # Fit the LSI model and perform dimensionality reduction
        X_train_ngramsTfidf_reduced = svd.fit_transform(X_train_ngramsTfidf)
        logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) on the training dataset", time.process_time())
        X_test_ngramsTfidf_reduced = svd.transform(X_test_ngramsTfidf)
        logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) on the test dataset", time.process_time())

        X_train = X_train_ngramsTfidf_reduced
        X_test = X_test_ngramsTfidf_reduced
    else:
        X_train = X_train_ngramsTfidf
        X_test = X_test_ngramsTfidf

    # # Extract features: offensive words
    # X_train_offensiveWordsTfidf_reduced, X_test_offensiveWordsTfidf_reduced, featureNames_offensiveWords =\
    #     extractFeatures_offensiveWords(docs_train, docs_test)

    # # Combine the n-grams with additional features:
    # X_train_combinedFeatures = np.concatenate((X_train_offensiveWordsTfidf_reduced,
    #                                            X_train_ngramsTfidf_reduced
    #                                            ), axis=1)
    # X_test_combinedFeatures = np.concatenate((X_test_offensiveWordsTfidf_reduced,
    #                                           X_test_ngramsTfidf_reduced
    #                                           ), axis=1)
    # featureNames_combinedFeatures = np.concatenate((featureNames_offensiveWords,
    #                                                 featureNames_ngrams
    #                                                 ), axis=0)

    ''' Build a classifier: Linear Support Vector classification
    - The underlying C implementation of LinearSVC uses a random number generator to select features when fitting the
    model. References:
        http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    '''
    clf = LinearSVC(random_state=42)
    # ↳ *dual=False* selects the algorithm to solve the primal optimization problem, as opposed to dual.
    # Prefer *dual=False* when n_samples > n_features. Source:
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    return X_train, X_test, clf, featureNames_ngrams


def extractFeatures_offensiveWords(docs_train, docs_test):
    """ This function performs the following tasks for the training and test datasets:
        1. Gets the counts of offensive words from the *countOffensiveWords()* function.
        2. Concatenates the count arrays for the desired Flame levels into X_train and X_test.
        3. Transforms the count matrix (NumPy array) to a normalized TF or TF-IDF representation.
        4. Performs dimensionality reduction on the normalized matrix using truncated SVD (aka LSA).
    Moreover, the function collects and returns the feature names (offensive expressions) for the desired Flame
    levels in the following format: “expression (flame level)”

    Important constants:
    DESIRED_FLAME_LEVELS: (Tuple: Ints) Desired flame levels. You can select any of the levels: 1, 2, 3, 4, and 5.
    """

    # Count the number of occurrences of all the offensive expressions in the training set and test set
    countsOfOffensiveWordsDict_train = countOffensiveWords(
        docs_train, "pickles/countsOfOffensiveWordsDict_train, <HASH>.pickle")
    countsOfOffensiveWordsDict_test = countOffensiveWords(
        docs_test, "pickles/countsOfOffensiveWordsDict_test, <HASH>.pickle")

    # Load the Flame Dictionary (to produce the list of feature names)
    flameDictionary, flameExpressionsDict = ProcessDataFiles.loadFlameDictionary()
    ''' ↳
        *flameDictionary*
            Keys:   (string) Expression
            Values: (int)    Flame level
        *flameExpressionsDict*
            Keys:   (int)           Flame level
            Values: (list: strings) Expressions
    '''

    # Log the min, max, and shape of the offensive words count arrays (just to make sure the pickles were loaded
    # correctly.
    for flameIndex in range(1, 6):
        array = countsOfOffensiveWordsDict_train[flameIndex]
        logger.debug("Flame level %d: min = %d | max = %-3d | shape = %s",
                     flameIndex, array.min(), array.max(), array.shape)
    for flameIndex in range(1, 6):
        array = countsOfOffensiveWordsDict_test[flameIndex]
        logger.debug("Flame level %d: min = %d | max = %-3d | shape = %s",
                     flameIndex, array.min(), array.max(), array.shape)

    # Create empty lists
    arraysList_train = []
    arraysList_test = []
    featureNames_offensiveWords = []

    # Concatenate the counts NumPy arrays and the feature names for the desired Flame levels
    DESIRED_FLAME_LEVELS = (1, 2, 3, 4, 5)
    for flameIndex in DESIRED_FLAME_LEVELS:
        arraysList_train.append(countsOfOffensiveWordsDict_train[flameIndex])
        arraysList_test.append(countsOfOffensiveWordsDict_test[flameIndex])
        # Add the expressions to the list of feature names in the form: “expression (flame level)”
        for expression in flameExpressionsDict[flameIndex]:
            featureNames_offensiveWords.append("{} ({})".format(expression, flameIndex))
    X_train_offensiveWordsCounts = np.concatenate(arraysList_train, axis=1)
    X_test_offensiveWordsCounts = np.concatenate(arraysList_test, axis=1)

    # • Transform the count matrix (NumPy array) to a normalized TF or TF-IDF representation
    # Build a TF-IDF transformer object
    tfidfTransformer = TfidfTransformer(norm='l2', use_idf=False, sublinear_tf=False)
    # ↳ With these parameters, the transformer does not make any changes: norm=None, use_idf=False, sublinear_tf=False
    ''' ↳ With normalization, each row (= author) is normalized to have a sum of absolute values / squares equal to 1.
    L^1-norm: Sum of the absolute value of the numbers (here, TF or TF-IDF of the offensive expressions)
    L^2-norm: Sum of the square         of the numbers ”...
    More info: http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
    '''
    # Fit and transform
    X_train_offensiveWordsTfidf = tfidfTransformer.fit_transform(X_train_offensiveWordsCounts)
    X_test_offensiveWordsTfidf = tfidfTransformer.transform(X_test_offensiveWordsCounts)

    # • Dimensionality reduction using truncated SVD (aka LSA)
    # Build a truncated SVD (LSA) transformer object
    svd_offensiveWords = TruncatedSVD(n_components=10, random_state=42)
    # Fit the LSI model and perform dimensionality reduction
    X_train_offensiveWordsTfidf_reduced = svd_offensiveWords.fit_transform(X_train_offensiveWordsTfidf)
    logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) in *extractFeatures_offensiveWords()* on "
                "the training dataset", time.process_time())
    X_test_offensiveWordsTfidf_reduced = svd_offensiveWords.transform(X_test_offensiveWordsTfidf)
    logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) in *extractFeatures_offensiveWords()* on "
                "the test dataset", time.process_time())

    return X_train_offensiveWordsTfidf_reduced, X_test_offensiveWordsTfidf_reduced, featureNames_offensiveWords


def countOffensiveWords(docs, picklePathPattern=None):
    """ This function counts the number of occurrences of all the expressions inside the Flame Dictionary.
        If the pickled results of the function (corresponding to the same input) already exists, the function is
        bypassed. If not, after the function is done, the results are stored as pickles.

        Inputs:
        - docs:       (list: strings) List of documents. Each row represents an author and contains one string.
        - picklePathPattern: (string) The path pattern for the pickle. This needs to include “<HASH>”, which will be
        replaced with the hash of the input of the function. Refer to docstrings of the *generatePicklePath* function.

        Output:
        - countsOfExpressionsDict: A Python dictionary
            • Keys:   (int)         Flame level
            • Values: (NumPy array) Counts of occurrences of expressions in that Flame level. Each row
            represents an author, and each column represents an expression in the Flame level of the key.

        Note: List of expressions can be accessed by calling *ProcessDataFiles.loadFlameDictionary*.
    """

    picklePath = generatePicklePath(docs, picklePathPattern)

    # Bypass: If the pickled results already exist, load (unpickle) and return them and skip the rest of the function
    if (picklePath is not None) and os.path.isfile(picklePath):
        with open(picklePath, 'rb') as pickleInputFile:
            unpickledObject = pickle.load(pickleInputFile)
        logger.info('Function bypassed: The counts of offensive words was loaded from pickle "%s" instead.', picklePath)
        return unpickledObject

    # Load the Flame Dictionary
    # TODO %%: Prevent loading the dictionary every time...
    flameDictionary, flameExpressionsDict = ProcessDataFiles.loadFlameDictionary()
    ''' ↳
    *flameDictionary*
        Keys:   (string) Expression
        Values: (int)    Flame level
    *flameExpressionsDict*
        Keys:   (int)           Flame level
        Values: (list: strings) Expressions
    '''

    # keysDictView = flameDictionary.keys()
    # expressions = list(keysDictView)

    # Preprocess the merged tweets of authors
    preprocessedDocs = []  # Create an empty list
    for authorIndex, doc in enumerate(docs):
        preprocessedDocs.append(preprocessTweet(doc))

        # Log after preprocessing the merged tweets of every 200 authors
        if authorIndex % 200 == 0:
            logger.debug("@ %.2f seconds, progress: Preprocessed the tweets of authorIndex = %d",
                         time.process_time(), authorIndex)
    logger.info("@ %.2f seconds: Finished preprocessing the tweets in *countOffensiveWords()*",
                time.process_time())

    # Create a dictionary of five NumPy arrays full of zeros
    countsOfExpressionsDict = {}  # Create an empty dictionary
    for flameIndex in range(1, 6):
        countsOfExpressionsDict[flameIndex] = np.zeros((len(preprocessedDocs),
                                                        len(flameExpressionsDict[flameIndex])), dtype=int)

    # Compile regex patterns into regex objects for all expressions, and store them in five separate lists, based on
    # Flame level (similar to *flameExpressionsDict*).
    ''' - Most regex operations are available as module-level functions as well as methods on compiled
        regular expressions. The functions are shortcuts that don’t require you to compile a regex object first,
        but miss some fine-tuning parameters.
        - Compiling a regex pattern and storing the resulting regex object for reuse is more efficient when the
        expression will be used several times in a single program. Even though the most recent patterns passed to
        re.compile() and the module-level matching functions are cached, the size of this cache is limited.
        More info: https://docs.python.org/3/library/re.html#re.compile
        Here, we are dealing with 2,600+ expressions, so the built-in cache cannot help. Storing the regex objects,
        decreased the processing time of each Author from 1.6 seconds to 0.7 seconds (on SaMaN-Laptop).
    '''
    ''' - In Python code, Regular Expressions will often be written using the raw string notation (r"text").
        Without it, every backslash in a regular expression would have to be prefixed with another one to escape it.
        - The shorthand \b matches a word boundary, without consuming any characters. Word boundary characters
        include space, . ! " ' - * and much more.
        - Some examples of matches of the /\bWORD\b/ pattern: WORD's, prefix-WORD, WORD-suffix, "WORD".
        %% TODO: To increase the performance of regex:
            1. I can combine the patterns using | for all expressions of the same level of Flame.
            https://stackoverflow.com/questions/1782586/speed-of-many-regular-expressions-in-python#comment1669596_1782712
            2. I can first use str.find to find potential matches, and then check those matches with regex. 
    '''
    regexObjectsDict = {1: [], 2: [], 3: [], 4: [], 5: []}  # Create a dictionary of 5 empty lists
    for flameIndex in range(1, 6):
        for expression in flameExpressionsDict[flameIndex]:
            regexPattern = r'\b' + expression + r'\b'
            regexObject = re.compile(regexPattern, re.IGNORECASE)
            regexObjectsDict[flameIndex].append(regexObject)
    logger.info("@ %.2f seconds: Finished compiling the regex patterns into regex objects.",
                time.process_time())

    # Count the matches of each expression for each author
    for authorIndex, mergedTweetsOfAuthor in enumerate(preprocessedDocs):
        for flameIndex in range(1, 6):
            for expressionIndex in range(len(flameExpressionsDict[flameIndex])):
                # ↳ Note: We are assuming that the lists inside *flameExpressionsDict* have not been manipulated since
                # the lists inside *regexObjectsDict* were created.
                listOfMatches = regexObjectsDict[flameIndex][expressionIndex].findall(mergedTweetsOfAuthor)
                count = len(listOfMatches)
                # count = mergedTweetsOfAuthor.count(expression)
                countsOfExpressionsDict[flameIndex][authorIndex, expressionIndex] = count

        # Log after counting the offensive words for every 100 authors
        if authorIndex % 100 == 0:
            logger.debug("@ %.2f seconds, progress: Counted (regex) the offensive words for authorIndex = %d",
                         time.process_time(), authorIndex)

    logger.info("@ %.2f seconds: Finished counting the occurrences of offensive words", time.process_time())

    # Pickle the output variable
    if picklePath is not None:
        # Create the directory if it does not exist.
        os.makedirs(os.path.dirname(picklePath), exist_ok=True)
        # Pickle
        with open(picklePath, 'wb') as pickleOutputFile:
            pickle.dump(countsOfExpressionsDict, pickleOutputFile)
        logger.info('The counts of offensive words was pickled to: "%s"', picklePath)

    return countsOfExpressionsDict


def generatePicklePath(inputObject, picklePathPattern):
    """ This function generates a “pickle path” to store the pickled output of a function, based on its two inputs:
            1. An object, which is the input of that function.
            2. A string, which holds the pattern of the pickle path.
        Here's how:
        1. *inputObject* is an object with any type. We calculate its SHA1 digest (hash value) encoded with the
        Base 32 encoding. The result is a 32-character, upper-case string (160 ÷ 5 = 32).
        2. *picklePathPattern* is a pattern of the pickle path, containing the placeholder sub-string, “<HASH>”.
        We replace this sub-string with the hash calculated in step 1, and return the result.

        Remarks:
        - If *picklePathPattern* is None, the function returns None.
    """

    # Define the constant
    HASH_PLACEHOLDER = "<HASH>"

    # Bypass 1: If *picklePathPattern* is None, return None and skip the rest of the function.
    if picklePathPattern is None:
        return None

    # Bypass 2: If *picklePathPattern* does not contain the *HASH_PLACEHOLDER* substring, raise an exception.
    if HASH_PLACEHOLDER not in picklePathPattern:
        raise ValueError('The pickle path pattern should contain the hash placeholder, "%s".' % HASH_PLACEHOLDER)
        # ↳ This is printf-style String Formatting.

    # Convert the input object to a *bytes* object (the pickled representation of the input object as a *bytes* object)
    inputObjectAsBytes = pickle.dumps(inputObject)
    # ↳ An inferior alternative could be *str(inputObject).encode("utf-8")*

    # Create a hash object that uses the SHA1 algorithm
    hashObject = hashlib.sha1()

    # Update the hash object with the *bytes* object. This will calculate the hash value.
    hashObject.update(inputObjectAsBytes)

    ''' • Get a digest (hash value) suitable for filenames—alpha-numeric, case insensitive, and
        relatively short: Base 32
        - The SHA1 algorithm produces a 160-bit (20-Byte) digest (hash value).
        - *hashObject.hexdigest()* returns the digest (hash value) as a string object, containing only hexadecimal
        digits. Each hexadecimal (also “base 16”, or “hex”) digit represents four binary digits (bits). As a result,
        a SHA1 hash represented as hex will have a length of 160 ÷ 4 = 40 characters.
        - *hashObject.digest()* returns the digest (hash value) as a *bytes* object.
        - *base64.b32encode()* encodes a *bytes* object using the Base32 encoding scheme—specified in RFC 3548—and
        returns the encoded *bytes* object.
        - The Base 32 encoding is case insensitive, and uses an alphabet of A–Z followed by 2–7 (32 characters).
        More info: https://tools.ietf.org/html/rfc3548#section-5
        Each Base 32 character represents 5 bits (2^5 = 32). As a result, a SHA1 hash represented as Base 32 will have
        a lenght of 160 ÷ 5 = 32 characters.
    '''
    hashValueAsBytes = hashObject.digest()
    hashValueAsBase32EncodedBytes = base64.b32encode(hashValueAsBytes)
    hashValueAsBase32EncodedString = hashValueAsBase32EncodedBytes.decode()
    # ↳ *.decode()* returns a string decoded from the given bytes. The default encoding is "utf-8".

    # Replace the *HASH_PLACEHOLDER* sub-string in path pattern with the Base 32 encoded digest (hash value)
    picklePath = picklePathPattern.replace(HASH_PLACEHOLDER, hashValueAsBase32EncodedString)

    return picklePath


def hexHashObject(inputObject):
    """ This function gets an object with any type as input, and returns its SHA1 digest (hash value) as a string,
        containing 40 hexadecimal digits.
    """

    # Convert the input object to a *bytes* object (the pickled representation of the input object as a *bytes* object)
    inputObjectAsBytes = pickle.dumps(inputObject)
    # ↳ An inferior alternative could be *str(inputObject).encode("utf-8")*

    # Create a hash object that uses the SHA1 algorithm
    hashObject = hashlib.sha1()

    # Update the hash object with the *bytes* object. This will calculate the hash value.
    hashObject.update(inputObjectAsBytes)

    # Get the hexadecimal digest (hash value)
    hexHashValue = hashObject.hexdigest()

    return hexHashValue


def crossValidateModel(clf, X_train, y_train):
    """ This function evaluates the classification model by k-fold cross-validation.
        The model is trained and tested k times, and all the scores are reported.
    """

    # Build a stratified k-fold cross-validator object
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    ''' Evaluate the score by cross-validation
        This fits the classification model on the training data, according to the cross-validator
        and reports the scores.
        Alternative: sklearn.model_selection.cross_validate
    '''
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=skf)

    logger.info("@ %.2f seconds: Cross-validation finished", time.process_time())

    # Log the cross-validation scores, the mean score and the 95% confidence interval, according to:
    # http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
    # https://en.wikipedia.org/wiki/Standard_error#Assumptions_and_usage
    logger.info("Scores = %s", scores)
    logger.info("%%Accuracy: %0.2f (±%0.2f)" % (scores.mean()*100, scores.std()*2*100))
    # ↳ https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html


def trainAndTestModel(clf, X_train, y_train, X_test, y_test):
    """ This function trains the classifier on the training set,
        predicts the classes on the test set using the trained model,
        and evaluates the accuracy of the model by comparing it to the truth of the test set.
    """

    # Fit the classification model on the whole training set (as opposed to cross-validation)
    clf.fit(X_train, y_train)

    ''' Predict the outcome on the test set
        Note that the clf classifier has already been fit on the training data.
    '''
    y_predicted = clf.predict(X_test)

    logger.info("@ %.2f seconds: Finished training the model and predicting class labels for the test set",
                time.process_time())

    # Simple evaluation using numpy.mean
    logger.info("numpy.mean %%Accuracy: %f", numpy.mean(y_predicted == y_test) * 100)

    # Log the classification report
    logger.info("Classification report:\n%s", metrics.classification_report(y_test, y_predicted))

    # Log the confusion matrix
    confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
    logger.info("Confusion matrix:\n%s", confusionMatrix)

    # %%%% TEMP: No plot!
    # # Plot the confusion matrix
    # plt.matshow(confusionMatrix)
    # plt.set_cmap('jet')
    # plt.show()


def trainModelAndPredict(clf, X_train, y_train, X_test, authorIDs_test, presetKey,
                         writeToXmlFiles=True, xmlsDestinationMainDirectory=None, ):
    """ This function is used only in **TIRA** evaluation.
        The difference between *trainModelAndPredict* and the *trainAndTestModel* function is that this function
        does not get *y_test* as an input, and hence, does not evaluate the accuracy of the model. Instead, it gets the
        Author IDs of the test dataset and writes the predictions as XML files for out-sourced evaluation.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'datasetName': 'PAN 2018 English',
                                            'languageCode': 'en',
                                            },
                          'PAN18_Spanish': {'datasetName': 'PAN 2018 Spanish',
                                            'languageCode': 'es',
                                            },
                          'PAN18_Arabic': {'datasetName': 'PAN 2018 Arabic',
                                            'languageCode': 'ar',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[presetKey]

    # Fit the classification model on the whole training set (as opposed to cross-validation)
    clf.fit(X_train, y_train)

    ''' Predict the outcome on the test set
        Note that the clf classifier has already been fit on the training data.
    '''
    y_predicted = clf.predict(X_test)

    logger.info("@ %.2f seconds: Finished training the model and predicting class labels for the test set",
                time.process_time())

    if writeToXmlFiles:
        logger.info("Writing the predictions to XML files.")
        ProcessDataFiles.writePredictionsToXmls(authorIDs_test, y_predicted,
                                                xmlsDestinationMainDirectory, PRESET['languageCode'])


def rankImportanceOfFeatures(clf, featureNames, writeToFile):
    """ This function ranks the features based on their importance in the classification model—absolute feature
    weight. It then writes the rankings to a CSV file (optional), and plots a number of top-ranking features and
    their weights.

    Inputs:
    - *clf*:          (LinearSVC)     A classifier object. This classifier should be trained.
    - *featureNames*: (list: strings) Feature names. You can get these from the vectorizer, when it is fit on the
    training set. For instance, the *vocabulary_* attribute of the vectorizers in scikit-learn.

    Returns:
    - *sortedFeatureWeights* (list: floats)  Feature weights, sorted by their absolute value in descending order.
    - *sortedFeatureNames*   (list: strings) Feature names, corresponding to the *sortedFeatureWeights* list.
    """

    # The NumPy array of feature weights (coefficients in the primal problem)
    featureWeights = list(clf.coef_.flatten())
    # ↳ *clf.coef_* is a NumPy array with the shape (1, n), where n is the number of features. *.flatten()*—a NumPy
    # function—collapses this array into one dimension, hence, giving an array with the shape (n,). We then convert
    # this array into a list.

    listOfTuples = list(zip(featureWeights, featureNames))
    # ↳ *zip()* makes an iterator that can be called only once. In order to reuse it, we convert it into a list.
    # Here, the result would be a list of n tuples, n being the number of features: (featureWeight, featureName)

    # Sort the list of tuples based on the absolute value of the feature weights in descending order.
    sortedListOfTuples = sorted(listOfTuples, key=lambda tuple: abs(tuple[0]), reverse=True)
    # ↳ *sorted* sorts the items in an iterable based on a *key* function (optional), and returns a list.

    # Split the sorted list of tuples into two lists
    sortedFeatureWeights, sortedFeatureNames = [list(a) for a in zip(*sortedListOfTuples)]
    # ↳ - zip(*sortedListOfTuples) returns an iterator of two tuples: (featureWeight1, ...) and (featureName1, ...)
    # - List comprehension: *list(a) for a in* is used to convert those two tuples into lists.
    # - An asterisk (*) denotes “iterable unpacking”.

    # Write the rankings to a CSV file
    if writeToFile:
        ProcessDataFiles.writeFeatureImportanceRankingsToCSV(sortedFeatureWeights, sortedFeatureNames,
                                                             logFilePath=logger.handlers[1].baseFilename)

    # Define constant: Number of top ranking features to plot
    PLOT_TOP = 30

    plt.barh(range(PLOT_TOP), sortedFeatureWeights[:PLOT_TOP], align='center')
    plt.yticks(range(PLOT_TOP), sortedFeatureNames[:PLOT_TOP])
    plt.xlabel("Feature weight")
    plt.ylabel("Feature name")
    plt.title("Top %d features based on absolute feature weight" % PLOT_TOP)

    # Flip the y axis
    plt.ylim(plt.ylim()[::-1])
    # ↳ plt.ylim() gets the current y-limits as a tuple. [::-1] reverses the tuple. plt.ylim(...) sets the new y-limits.
    # [start:end:step] is the “extended slicing” syntax. Some more info available at:
    # https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy

    plt.show()

    return sortedFeatureWeights, sortedFeatureNames


def main_Development():
    """ This function is the “main” function for development.
        Every time the script runs, it will call this function.
    """

    for presetKey in ("PAN18_English", "PAN18_Spanish", "PAN18_Arabic"):

        logger.info("Running main_Development() for preset: %s", presetKey)

        docs_train, docs_test, y_train, y_test = loadDatasets_Development(presetKey)
        X_train, X_test, clf, featureNames = extractFeatures(docs_train, docs_test, presetKey)
        crossValidateModel(clf, X_train, y_train)
        trainAndTestModel(clf, X_train, y_train, X_test, y_test)

        # %%%
        # rankImportanceOfFeatures(clf, featureNames, True)

        # Log run time
        logger.info("@ %.2f seconds: Run finished\n", time.process_time())


def main_TiraEvaluation():
    """ This... %%

    """

    logger.info("sys.argv = %s", sys.argv)

    ''' Parse the command line arguments
    According to PAN, the submitted script will be executed via command line calls with the following format:
        interpreter.exe script.py -c $inputDataset -o $outputDir
    
    For local testing on SaMaN-Laptop, you can use the following command (replace $inputDataset and $outputDir):
    C:/Users/Saman/Miniconda3/python.exe C:/Users/Saman/PycharmProjects/PAN18_AuthorProfiling/TrainModel.py -c $inputDataset -o $outputDir
    '''
    # Build a parser
    commandLineArgumentParser = argparse.ArgumentParser()
    commandLineArgumentParser.add_argument("-c")
    commandLineArgumentParser.add_argument("-o")

    # Parse arguments
    commandLineArguments = commandLineArgumentParser.parse_args()
    testDatasetMainDirectory = commandLineArguments.c
    # ↳ This will be ignored for now.
    predictionXmlsDestinationMainDirectory = commandLineArguments.o

    # # TEMP (TIRA): For local testing on SaMaN-Laptop
    # testDatasetMainDirectory = "data/TiraDummy/testDirectory"
    # predictionXmlsDestinationMainDirectory = "data/TiraDummy/predictionXmls"

    # # TEMP (TIRA): For local testing on TIRA
    # testDatasetMainDirectory = "E:/author-profiling/pan18-author-profiling-training-dataset-2018-02-27"
    # outputFolderName = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    # predictionXmlsDestinationMainDirectory = os.path.join("output", outputFolderName)
    # os.makedirs(predictionXmlsDestinationMainDirectory, exist_ok=True)  # Create the directory if it does not exist

    logger.info("testDatasetMainDirectory = %s", testDatasetMainDirectory)
    logger.info("predictionXmlsDestinationMainDirectory = %s\n", predictionXmlsDestinationMainDirectory)

    for presetKey in ("PAN18_English", "PAN18_Spanish", "PAN18_Arabic"):
        logger.info("Running main_TiraEvaluation() for preset: %s", presetKey)

        docs_train, docs_test, y_train, authorIDs_test =\
            loadDatasets_TiraEvaluation(testDatasetMainDirectory, presetKey)
        # ↳ There is no *y_test* because the truth of the test dataset will not be provided to the participants.

        # # TEMP (TIRA): For fast debugging and testing
        # docs_train = docs_train[:100]
        # docs_test = docs_test[:100]
        # y_train = y_train[:100]
        # authorIDs_test = authorIDs_test[:100]

        X_train, X_test, clf, featureNames = extractFeatures(docs_train, docs_test, presetKey)
        crossValidateModel(clf, X_train, y_train)
        trainModelAndPredict(clf, X_train, y_train, X_test, authorIDs_test, presetKey,
                             True, predictionXmlsDestinationMainDirectory)

        # Log run time
        logger.info("@ %.2f seconds: Run finished\n", time.process_time())


''' The following lines will be executed only if this .py file is run as a script,
    and not if it is imported as a module.
    • __name__ is one of the import-related module attributes, which holds the name of the module.
    • A module's __name__ is set to "__main__" when it is running in
    the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    logger = configureRootLogger()
    checkSystemInfo()
    main_Development()
    # main_TiraEvaluation()
