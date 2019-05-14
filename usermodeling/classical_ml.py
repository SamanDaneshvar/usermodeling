"""Perform classical machine learning experiments on the ASI dataset

This script contains gender and age classification experiments on the ASI dataset using the SVM algorithm
"""

import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from usermodeling import datasets
from usermodeling.pan18ap_classical_ml import preprocess_tweet
from usermodeling.utils import my_utils


def load_split_asi_dataset(task):
    """Load and split the ASI (Advanced Symbolics Inc.) dataset

    - Load the preprocessed dataset (URLs and username mentions removed, repeated characters normalized,
        lowercased, etc.)
    - Split the dataset into balanced (stratified) training (60%), validation (20%), and test (20%) sets
    """

    LABELS_XML_PATH = 'data/Advanced Symbolics/Labels.xml'
    TWEETS_XMLS_DIR = 'data/Advanced Symbolics/Tweets'

    # Based on the task (gender or age classification), when loading the ASI dataset, the dataset will be stratified
    # on the corresponding label
    if task == 'gender':
        stratified_subset = 'genders'
    elif task == 'age':
        stratified_subset = 'ages'
    else:
        raise ValueError('The *task* parameter is expected to be "gender" or "age", not "%s".' % task)

    # Load the raw texts and the labels from the files into lists
    user_ids, processed_merged_tweets, text_genders, text_ages = \
        datasets.asi.load(LABELS_XML_PATH, TWEETS_XMLS_DIR, stratified_subset=stratified_subset)

    # Based on the task, select the labels to work with
    if task == 'gender':
        labels = text_genders
    elif task == 'age':
        labels = text_ages
    else:
        raise ValueError('The *task* parameter is expected to be "gender" or "age", not "%s".' % task)

    # Split the raw dataset into balanced (stratified) training+validation and test sets (split 20% for test set)
    (docs_trainval, docs_test,
     y_trainval, y_test,
     user_ids_trainval, user_ids_test) = train_test_split(processed_merged_tweets, labels, user_ids,
                                                          test_size=0.2, random_state=42, stratify=labels)
    # ↳ *stratify=labels* selects a balanced sample from the data, with the same class proportion as
    #   the *labels* list.

    # Split the raw training+validation dataset into balanced (stratified) training and validation sets
    # Note: 20% (validation set) is 25% of 80% (training+validation), hence the *test_size=0.25* option.
    (docs_train, docs_val,
     y_train, y_val,
     user_ids_train, user_ids_val) = train_test_split(docs_trainval, y_trainval, user_ids_trainval,
                                                      test_size=0.25, random_state=42, stratify=y_trainval)
    # ↳ Note: The array-like object given to the *stratify* option should have the same number of samples as the inputs.

    return docs_train, docs_val, docs_test, y_train, y_val, y_test


def load_pan18ap_test_corpus():
    """Load and pre-process the official test corpus of PAN 2018 (only English)

    - Load the English test corpus of the PAN 2018 Author Profiling task
    - Pre-process the raw text (replace URLs, etc.)

    This is used as a second test set for age classification experiments on the ASI dataset.
    """

    XMLS_DIRECTORY = 'data/PAN 2018, Author Profiling - Test Corpus/en/text'
    TRUTH_PATH = 'data/PAN 2018, Author Profiling - Test Corpus/en/en.txt'

    # Load the raw texts and the labels (truths) from the files into lists
    merged_tweets, text_labels, user_ids, _ignore = \
        datasets.process_data_files.load_pan_data(XMLS_DIRECTORY, TRUTH_PATH)

    # Process the merged tweets using NLTK's tweet tokenizer to replace repeated characters,
    # and remove URLs and @Username mentions
    processed_merged_tweets = []  # Create an empty list
    for merged_tweets_of_author in merged_tweets:
        processed_merged_tweets.append(preprocess_tweet(merged_tweets_of_author, replacement_tags=False))

    return processed_merged_tweets, text_labels  # docs, y


def extract_features_gender(docs_train, docs_val, docs_test_asi, docs_test_pan18ap, lsa=True):
    """Extract features

    - Build a transformer (vectorizer) pipeline
    - Fit the transformer to the training set (learns vocabulary and idf)
    - Transform the training set and the validation/test sets to their TF-IDF matrix representation
    - Perform dimensionality reduction using LSA on the TF-IDF matrices (optional)
    """

    # Build a vectorizer that splits strings into sequences of 1 to 3 words
    word_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                      analyzer='word', ngram_range=(1, 3),
                                      min_df=2, use_idf=True, sublinear_tf=True)
    # Build a vectorizer that splits strings into sequences of 3 to 5 characters
    char_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                      analyzer='char', ngram_range=(3, 5),
                                      min_df=2, use_idf=True, sublinear_tf=True)

    # Build a transformer (vectorizer) pipeline using the previous analyzers
    # *FeatureUnion* concatenates results of multiple transformer objects
    ngrams_vectorizer = Pipeline([('feats', FeatureUnion([('word_ngram', word_vectorizer),
                                                          ('char_ngram', char_vectorizer)]))])

    # Fit (learn vocabulary and IDF) and transform (transform documents to the TF-IDF matrix) the training set
    x_train_ngrams_tfidf = ngrams_vectorizer.fit_transform(docs_train)
    '''
    ↳ Check the following attributes of each of the transformers (analyzers)—*word_vectorizer* and *char_vectorizer*:
    vocabulary_ : dict. A mapping of terms to feature indices.
    stop_words_ : set. Terms that were ignored
    '''
    logger.info('@ %.2f seconds: Finished fit_transforming the training dataset', time.process_time())

    feature_names_ngrams = [word_vectorizer.vocabulary_, char_vectorizer.vocabulary_]

    # Vectorize each validation/test set
    # Extract the features of the validation/test sets (transform test documents to the TF-IDF matrix)
    # Only transform is called on the transformer (vectorizer), because it has already been fit to the training set.
    x_val_ngrams_tfidf = ngrams_vectorizer.transform(docs_val)
    logger.info('@ %.2f seconds: Finished transforming the validation set', time.process_time())
    x_test_asi_ngrams_tfidf = ngrams_vectorizer.transform(docs_test_asi)
    logger.info('@ %.2f seconds: Finished transforming the ASI test set', time.process_time())
    x_test_pan18ap_ngrams_tfidf = ngrams_vectorizer.transform(docs_test_pan18ap)
    logger.info('@ %.2f seconds: Finished transforming the PAN18AP test set', time.process_time())

    logger.info('Word & character ngrams .shape = {training: %s | validation: %s | test_asi: %s, test_pan18ap: %s}',
                x_train_ngrams_tfidf.shape, x_val_ngrams_tfidf.shape,
                x_test_asi_ngrams_tfidf.shape, x_test_pan18ap_ngrams_tfidf.shape)

    # • Dimensionality reduction using truncated SVD (aka LSA)
    if lsa:
        # Build a truncated SVD (LSA) transformer object
        svd = TruncatedSVD(n_components=300, random_state=43)
        # Fit the LSA model and perform dimensionality reduction on the training set
        x_train_ngrams_tfidf_reduced = svd.fit_transform(x_train_ngrams_tfidf)
        logger.info('@ %.2f seconds: Finished dimensionality reduction (LSA) on the training set',
                    time.process_time())
        # Perform dimensionality reduction on the validation/test sets
        # Note that the SVD (LSA) transformer is already fit on the training set
        x_val_ngrams_tfidf_reduced = svd.transform(x_val_ngrams_tfidf)
        logger.info('@ %.2f seconds: Finished dimensionality reduction (LSA) on the validation set',
                    time.process_time())
        x_test_asi_ngrams_tfidf_reduced = svd.transform(x_test_asi_ngrams_tfidf)
        logger.info('@ %.2f seconds: Finished dimensionality reduction (LSA) on the ASI test set',
                    time.process_time())
        x_test_pan18ap_ngrams_tfidf_reduced = svd.transform(x_test_pan18ap_ngrams_tfidf)
        logger.info('@ %.2f seconds: Finished dimensionality reduction (LSA) on the PAN18AP test set',
                    time.process_time())

        x_train = x_train_ngrams_tfidf_reduced
        x_val = x_val_ngrams_tfidf_reduced
        x_test_asi = x_test_asi_ngrams_tfidf_reduced
        x_test_pan18ap = x_test_pan18ap_ngrams_tfidf_reduced
    else:
        x_train = x_train_ngrams_tfidf
        x_val = x_val_ngrams_tfidf
        x_test_asi = x_test_asi_ngrams_tfidf
        x_test_pan18ap = x_test_pan18ap_ngrams_tfidf

    return x_train, x_val, x_test_asi, x_test_pan18ap, feature_names_ngrams


def extract_features_age(docs_train, docs_val, docs_test, lsa=True):
    """%%%"""
    pass


def def_train_model(x_train, y_train):
    """Build a classifier and train it

    Args:
        x_train: Vector representation of the training set
        y_train: Labels of the training set

    Returns:
        clf: The trained classifier
    """

    # Build a classifier: Linear Support Vector classification
    # - The underlying C implementation of LinearSVC uses a random number generator to select features when fitting the
    # model.
    # References:
    #     http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    #     http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    clf = LinearSVC(random_state=42)
    # ↳ *dual=False* selects the algorithm to solve the primal optimization problem, as opposed to dual.
    # Prefer *dual=False* when n_samples > n_features. Source:
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    # Fit the classification model on the training set
    clf.fit(x_train, y_train)

    logger.info('@ %.2f seconds: Finished training the model on the training set', time.process_time())

    return clf


def test_model(trained_clf, x_test, y_test, TEST_SET_LABEL='test'):
    """Test the trained model

    Predict the classes on the validation/test set using the trained model,
    and evaluate the accuracy of the model by comparing it to the truth of the test set.
    """

    # Predict the labels for the test set
    y_predicted = trained_clf.predict(x_test)

    logger.info('@ %.2f seconds: Finished predicting the labels of the %s set',
                time.process_time(), TEST_SET_LABEL)

    # Simple evaluation using numpy.mean
    logger.info('>> Mean accuracy: %f', np.mean(y_predicted == y_test))

    # Log the classification report
    classification_report = metrics.classification_report(y_test, y_predicted)
    logger.info('>> Classification report:')
    # Break the multi-line string into a list of lines and pass the lines one by one to the logger
    lines = classification_report.splitlines()
    for line in lines:
        logger.info(line)

    # Log the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
    logger.info('>> Confusion matrix:')
    # Convert the ndarray object to string, split it into a list of lines and pass the lines one by one to the logger
    lines = str(confusion_matrix).splitlines()
    for line in lines:
        logger.info(line)


def main_gender():
    """The main function for gender classification experiments"""

    logger.info('Experiment notes: --')

    docs_train, docs_val, docs_test_asi, y_train, y_val, y_test_asi = load_split_asi_dataset(task='gender')
    docs_test_pan18ap, y_test_pan18ap = load_pan18ap_test_corpus()

    (x_train, x_val, x_test_asi, x_test_pan18ap,
     feature_names_ngrams) = extract_features_gender(docs_train, docs_val, docs_test_asi, docs_test_pan18ap, lsa=True)

    trained_clf = def_train_model(x_train, y_train)

    test_model(trained_clf, x_val, y_val, TEST_SET_LABEL='validation')
    test_model(trained_clf, x_test_asi, y_test_asi, TEST_SET_LABEL='ASI test')
    test_model(trained_clf, x_test_pan18ap, y_test_pan18ap, TEST_SET_LABEL='PAN18AP test')

    # Log run time
    logger.info('@ %.2f seconds: Run finished', time.process_time())


def main_age():
    """The main function for age classification experiments %%%"""


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
    main_gender()
