"""ASI dataset"""

import logging
import os
import time
from xml.etree import ElementTree as ET

from usermodeling.utils import my_utils


def load(labels_xml_path, tweets_xmls_dir):
    """Load the ASI dataset from pre-processed XML files.

    Args:
        labels_xml_path: The path to the labels XML file.
        tweets_xmls_dir: The directory where XML files of users (all tweets of the user) reside.

    Returns:
         user_ids <list of str>: User IDs
         merged_tweets_of_users <list of str>: Merged tweets of users (in the same order as *user_ids*)
         genders <list of str>: Gender labels of users (in the same order as *user_ids*)
         ages <list of str>: Age labels (in the same order as *user_ids*)
    """

    user_ids, genders, ages = _load_labels(labels_xml_path)
    tweets_of_users = _load_tweets(user_ids, tweets_xmls_dir)

    # Merge the tweets of each user into a single string
    merged_tweets_of_users = []
    for tweets_of_user in tweets_of_users:
        merged_tweets_of_user = ' <TweetBorder> '.join(tweets_of_user)
        merged_tweets_of_users.append(merged_tweets_of_user)

    logger.info('TEMP!')

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


def _load_tweets(user_ids, xmls_dir):
    """Load the tweets of the ASI dataset from XML files.

    Args:
        user_ids <list of str>: List of user IDs to be loaded.
        xmls_dir <str>: The directory where XML files of users (all tweets of the user) reside.

    Returns:
        <list of list of str>: List of tweets for each user. Each string is a tweet of a user.
    """

    tweets_of_users = []

    # Iterate over the user IDs
    for i, user_id in enumerate(user_ids):
        XML_PATH = os.path.join(xmls_dir, user_id + '.xml')

        # Read the XML file and parse it into a tree
        # Parser is explicitly defined to ensure UTF-8 encoding.
        tree = ET.parse(XML_PATH, parser=ET.XMLParser(encoding='utf-8'))
        L0_root = tree.getroot()

        tweets_of_this_user = []

        # Iterate over the tweets within the parsed XML file
        for L1_tweet in L0_root:
            # *Element.find()* finds the first child of the current element with a particular tag.
            text = L1_tweet.find('text').text
            tweets_of_this_user.append(text)

        tweets_of_users.append(tweets_of_this_user)

        # Log the progress
        if (i + 1) % 1000 == 0:
            logger.info('@ %.2f seconds: Finished loading the tweets of user #%s',
                        time.process_time(), format(i + 1, ',d'))

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

    # TODO
    logger, RUN_TIMESTAMP = my_utils.configure_root_logger(1)
    my_utils.set_working_directory(2)
    load('data/Advanced Symbolics/Labels.xml', 'data/Advanced Symbolics/Tweets')
