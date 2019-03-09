"""Prepare the Advanced Symbolics (ASI) dataset"""

import csv
from datetime import datetime
import fnmatch
import os
import time

import numpy as np

from usermodeling.utils import my_utils
from usermodeling.classical_ml import preprocess_tweet


class Dataset:
    """Dataset class"""

    def __init__(self):
        self.users = []

    def load_tweets(self, CSVS_ROOT_DIR):
        """Load all tweets from CSV files

        Args:
            CSVS_ROOT_DIR: The root directory containing CSV files of users (all tweets of user)
        """

        # Go through every file in the directory and its subdirectories,
        # and load the tweets into Tweet objects.
        for root, dirs, files in os.walk(CSVS_ROOT_DIR):
            for filename in files:
                if fnmatch.fnmatch(filename, '*.csv'):
                    csv_file_path = os.path.join(root, filename)
                    user_id = filename[:-4]
                    # Construct a *User* object
                    this_user = User(user_id)
                    # Get hold of this user in the *users* list
                    self.users.append(this_user)
                    # Go through the tweets in the CSV file
                    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        next(csv_reader)  # Skip the first row (header)
                        for row in csv_reader:
                            # Create a *Tweet* object
                            # The initializer of the Tweet class will also call the *add_tweet* method for the user
                            Tweet(this_user, row[0], row[1], row[2], row[3])

                    # Log the progress
                    if len(self.users) % 1000 == 0:
                        logger.info('@ %.2f seconds: Finished loading user #%s',
                                    time.process_time(), format(len(self.users), ',d'))

        # Log basic stats
        num_users = len(self.users)
        num_tweets = 0
        for user in self.users:
            num_tweets += len(user.get_tweets())

        logger.info('@ %.2f seconds: Finished loading the dataset: %s users and %s tweets',
                    time.process_time(), format(num_users, ',d'), format(num_tweets, ',d'))

    def preprocess_tweets(self):
        """Preprocess the tweets in the dataset"""

        for i, user in enumerate(self.users):
            for tweet in user.get_tweets():
                tweet.preprocess()
            # Log the progress
            if (i + 1) % 100 == 0:
                logger.info('@ %.2f seconds: Finished preprocessing the tweets of user #%s',
                            time.process_time(), format(i + 1, ',d'))

        logger.info('@ %.2f seconds: Finished preprocessing the tweets', time.process_time())

    def drop_retweets(self, drop=True):
        """Remove all the tweets that are retweets from the dataset

        Args:
            drop: Boolean. When False, the method doubles as a counter without removing any of the retweets.

        Returns:
            A NumPy array containing the count of retweets for each user in the dataset
        """

        retweet_counts = []

        for user in self.users:
            retweet_count = 0
            for tweet in user.get_tweets():
                if tweet.is_retweet():
                    retweet_count += 1
                    if drop:
                        user.remove_tweet(tweet)
            retweet_counts.append(retweet_count)

        return np.array(retweet_counts)

    def get_user(self, user_id):
        """Find a user based on user ID

        The method returns the first User object from the dataset which has the given user ID.

        Args:
            user_id: A string containing the user ID

        Returns:
            A User object matching the given user ID
        """

        for user in self.users:
            if user.get_id() == user_id:
                return user

    def get_num_tweets(self):
        """Get number of tweets for all users in the dataset

        Returns:
            A NumPy array. Each item in the array is the count of tweets for each user in the dataset.
        """
        report = []
        for user in self.users:
            report.append(user.get_num_tweets())

        return np.array(report)

    def get_stats(self):
        """Produce statistics of the dataset"""

        # TODO
        # for user in self.users:


class User:
    """Twitter user class"""

    def __init__(self, user_id):  # Initializer (instance attributes)
        self.__id = user_id  # Private data field
        self.__tweets = []

        self.gender = ''
        self.age = ''

    def add_tweet(self, tweet):  # Instance method
        """Add a tweet object to the user"""
        self.__tweets.append(tweet)

    def remove_tweet(self, tweet):
        """Remove a tweet object from the user's tweets"""

        self.__tweets.remove(tweet)
        # ↳ *array.remove(x)* removes the first occurrence of x from the array.

    def get_id(self):
        """Getter (accessor) for user ID"""
        return self.__id

    def get_tweets(self):
        """Getter (accessor) for the user's tweets"""
        return self.__tweets

    def get_num_tweets(self):
        """Get number of tweets of the user"""
        return len(self.__tweets)


class Tweet:
    """Tweet class"""

    def __init__(self, user, datetime_string, language, original_text, url):
        self.__user = user
        user.add_tweet(self)

        # Remove the colon from the UTC offset (last six characters): '+00:00'
        datetime_string = datetime_string[:-5] + datetime_string[-5:-3] + datetime_string[-2:]
        # Parse the string into a *datetime* object
        self.datetime = datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S%z')

        self.language = language
        self.original_text = original_text
        self.text = ''
        self.url = url
        self.__retweet = ''

    def get_user(self):
        """Getter (accessor) for the user attribute"""
        return self.__user

    def is_retweet(self):
        """Getter (accessor) for the boolean *retweet* attribute"""
        return self.__retweet

    def preprocess(self):
        """Preprocess the tweet (only if it is not a retweet)

        - Determine whether the tweet is a retweet or not, and update its boolean *retweet* attribute accordingly.
        - If not a retweet, preprocess the tweet and update its *text* attribute with the processed text
        """

        # Determine whether the tweet is a retweet or not
        if self.original_text[:3] == 'RT ':
            self.__retweet = True
        else:
            self.__retweet = False
            # Preprocess the tweet
            self.text = preprocess_tweet(self.original_text)


def main():
    """The main function"""

    CSV_BATCHES_DIR = 'P:/2018-12-20_13-31-03 _ Batch 1'

    dataset = Dataset()  # Constructor
    dataset.load_tweets(CSV_BATCHES_DIR)
    dataset.preprocess_tweets()

    tweet_counts_before = dataset.get_num_tweets()

    retweet_counts = dataset.drop_retweets()
    logger.info('@ %.2f seconds: Finished removing the retweets', time.process_time())
    logger.info('A total of %s retweets dropped.', format(sum(retweet_counts), ',d'))

    tweet_counts_after = dataset.get_num_tweets()

    if np.array_equal(tweet_counts_before - retweet_counts, tweet_counts_after):
        logger.info('Success: The counts of tweets and retweets before and after drop checks out!')


    # dataset.get_stats()

    # Log run time
    logger.info("@ %.2f seconds: Run finished", time.process_time())


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to "__main__" when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    proj_dir = 'C:/Users/Saman/GitHub/usermodeling'
    logger, RUN_TIMESTAMP = my_utils.configure_root_logger(proj_dir)
    my_utils.set_working_directory(proj_dir)
    main()
