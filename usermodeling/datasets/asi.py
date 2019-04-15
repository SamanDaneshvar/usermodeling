"""ASI dataset"""

from collections import Counter
import logging
import os
import time
from xml.etree import ElementTree as ET

import numpy as np

from usermodeling.utils import my_utils


def load(labels_xml_path, tweets_xmls_dir, stratified_subset=None):
    """Load the ASI dataset from pre-processed XML files.

    Args:
        labels_xml_path: The path to the labels XML file.
        tweets_xmls_dir: The directory where XML files of users (all tweets of the user) reside.
        stratified_subset: If not None (default), the largest possible subset of the dataset is loaded, such that it is
            stratified over the given labels. The possible values are 'genders' and 'ages'.

    Returns:
         user_ids <list of str>: User IDs
         merged_tweets_of_users <list of str>: Merged tweets of users (in the same order as *user_ids*)
         genders <list of str>: Gender labels of users (in the same order as *user_ids*)
         ages <list of str>: Age labels (in the same order as *user_ids*)
    """

    user_ids, genders, ages = _load_labels(labels_xml_path)

    # TODO: TEMP (for quick debug)
    # user_ids = user_ids[:100]
    # genders = genders[:100]
    # ages = ages[:100]

    if stratified_subset == 'genders':
        user_ids, genders, ages = _stratify(user_ids, genders, ages, stratify=genders)
    elif stratified_subset == 'ages':
        user_ids, genders, ages = _stratify(user_ids, genders, ages, stratify=ages)

    if stratified_subset is not None:
        logger.info('Selected a random subset of the dataset, stratified on %s: %s users',
                    stratified_subset, format(len(user_ids), ',d'))

    tweets_of_users = _load_tweets(user_ids, tweets_xmls_dir)

    # Merge the tweets of each user into a single string
    merged_tweets_of_users = []
    for tweets_of_user in tweets_of_users:
        merged_tweets_of_user = ' <TweetBorder> '.join(tweets_of_user)
        merged_tweets_of_users.append(merged_tweets_of_user)

    logger.info('@ %.2f seconds: Finished loading the ASI dataset', time.process_time())

    return user_ids, merged_tweets_of_users, genders, ages


def _load_labels(xml_path):
    """Load the labels of the ASI dataset from XML file.

    Note: The location and race are left out. To also load these two labels, uncomment the related lines and add
    modify the *return* line.

    Args:
        xml_path: The path to the labels XML file.

    Returns:
        user_ids <List of str>: User IDs
        genders <List of str>: Gender labels ('female'/'male')
        ages <List of str>: Age labels ('[0,25)'/...)
    """

    user_ids = []
    genders = []
    ages = []
    # locations = []
    # races = []

    # Read the XML file and parse it into a tree
    # Parser is explicitly defined to ensure UTF-8 encoding.
    tree = ET.parse(xml_path, parser=ET.XMLParser(encoding='utf-8'))
    L0_root = tree.getroot()
    '''
    root is the root element of the parsed tree
    root[0], ..., root[m-1] are the children of root—elements one level below the root.
    root[0][0], ..., root[0][n-1] are the children of root[0].
    and so on.

    Each element has a tag, a dictionary of attributes, and sometimes some text:
        root[i][j].tag, .attrib, .text
        
    You can access a certain attribute of an element using *Element.get()*.
    '''

    # Iterate over the users within the parsed XML file.
    for L1_user in L0_root:
        user_id = L1_user.get('user_id')
        # *Element.find()* finds the first child of the current element with a particular tag.
        gender = L1_user.find('gender').text
        age = L1_user.find('age').text
        # location = L1_user.find('location').text
        # race = L1_user.find('race').text

        user_ids.append(user_id)
        genders.append(gender)
        ages.append(age)
        # locations.append(location)
        # races.append(race)

    logger.info('@ %.2f seconds: Finished loading the labels of the ASI dataset: %s users',
                time.process_time(), format(len(user_ids), ',d'))

    return user_ids, genders, ages


def _stratify(*arrays, stratify):
    """Return the largest possible subset of the dataset, such that it is stratified over the given labels.

    Note: The order of the items in *stratify* array is expected to be the same as that of *arrays*.

    Args:
        *arrays (arbitrary argument list): The arrays which will be stratified. All these arrays and the *stratify*
            array are expected to be of the same length.
        stratify (array-like): The labels over which the dataset will be stratified.

    Returns:
        The same number of lists as the *arrays arguments received, stratified over the *stratify* label.
    """

    # • Shuffle the arrays
    # Build the random indices. All arrays will be shuffle in the same orderr
    indices = np.arange(len(stratify))
    np.random.seed(42)
    np.random.shuffle(indices)
    # Convert *arrays* and *stratify* to an array. We need to do this to take advantage of the
    # *array[indices]* functionality below.
    arrays = np.array(arrays)
    stratify = np.array(stratify)
    # Shuffle the arrays in *arrays*
    for i in range(len(arrays)):
        arrays[i] = arrays[i][indices]
    # Shuffle *stratify*
    stratify = stratify[indices]

    # Find the minimum number of occurrences of the labels in the *stratify* list
    # In the largest possible subset of the dataset that can be stratified, we would have this many items of each label.
    label_counts_dict = Counter(stratify)
    min_label_count = min(label_counts_dict.values())

    # Create a dictionary with labels in *stratify* as keys and 0 as values.
    # We will use this to count how many samples with each label we have picked for the stratified subset so far,
    # to know when to stop.
    num_in_subset = {key:0 for key in label_counts_dict.keys()}

    # Initialize a list of empty lists with the same size as *arrays*
    # This will contain the stratified subset selected from the dataset
    stratified_arrays = []
    for i in range(len(arrays)):
        stratified_arrays.append([])

    for i, label in enumerate(stratify):
        if num_in_subset[label] >= min_label_count:
            # We have already picked the samples that we need for the subset. Ignore this sample.
            pass
        else:
            # Append the corresponding items in each of the arrays to the stratified arrays
            for array_index in range(len(arrays)):
                stratified_arrays[array_index].append(arrays[array_index][i])
            num_in_subset[label] += 1

    return stratified_arrays


def _load_tweets(user_ids, xmls_dir):
    """Load the tweets of the ASI dataset from XML files.

    Args:
        user_ids <list of str>: List of user IDs to be loaded.
        xmls_dir <str>: The directory where XML files of users (all tweets of the user) reside.

    Returns:
        <list of list of str>: List of tweets for each user. Each string is a tweet of a user.
    """

    tweets_of_users = []
    user_ids_with_missing_files = []

    # Iterate over the user IDs
    for i, user_id in enumerate(user_ids):
        XML_PATH = os.path.join(xmls_dir, user_id + '.xml')

        tweets_of_this_user = []

        # Check if the CSV file for this user exists.
        if os.path.isfile(XML_PATH):
            # Read the XML file and parse it into a tree
            # Parser is explicitly defined to ensure UTF-8 encoding.
            tree = ET.parse(XML_PATH, parser=ET.XMLParser(encoding='utf-8'))
            L0_root = tree.getroot()

            # Iterate over the tweets within the parsed XML file
            for L1_tweet in L0_root:
                # *Element.find()* finds the first child of the current element with a particular tag.
                text = L1_tweet.find('text').text
                tweets_of_this_user.append(text)
        else:
            # The CSV file for this user does not exist.
            # The user will be notified at the end of the function.
            user_ids_with_missing_files.append(user_id)

        tweets_of_users.append(tweets_of_this_user)
        # ↳ If the CSV file for the user is missing, the *tweets_of_this_user* list remains empty.
        #   We would still append that empty list to the *tweets_of_users* list to maintain the order of the users
        #   between that list and the *user_ids* list.

        # Log the progress
        if (i + 1) % 1000 == 0:
            logger.info('@ %.2f seconds: Finished loading the tweets of user #%s',
                        time.process_time(), format(i + 1, ',d'))

    if len(user_ids_with_missing_files) > 0:
        logger.error('Could not load the tweets of %s users due to missing CSV files: %s',
                     format(len(user_ids_with_missing_files), ',d'), user_ids_with_missing_files)

    return tweets_of_users

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

