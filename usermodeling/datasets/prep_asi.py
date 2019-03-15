"""Prepare the Advanced Symbolics (ASI) dataset"""

import csv
from datetime import datetime
import fnmatch
import os
import pickle
import time

import numpy as np

from usermodeling.utils import my_utils
from usermodeling.classical_ml import preprocess_tweet


class Dataset:
    """Dataset class"""

    def __init__(self):
        self.users = set()  # Create an empty set

    def add_user(self, user):
        """Add user object to the dataset"""
        # *Dataset.users* is a set
        self.users.add(user)

    def remove_user(self, user):
        """Remove a user object from the dataset"""
        self.users.remove(user)

    def load_labels_and_create_users(self, DEMOGRAPHICS_CSV_PATH, USERS_LIST_PATH):
        """Load the labels (demographics) of all users from a CSV file

        Args:
            DEMOGRAPHICS_CSV_PATH: The path of the CSV file containing the demographics of users
            USERS_LIST_PATH: The path of the text file containing the list of target users IDs. From the
                demographics CSV file, only the rows related to users that are in this list will be loaded.
        """

        logger.info('Loading the demographics of the ASI dataset from: "%s"', DEMOGRAPHICS_CSV_PATH)

        # Read the target users into a set. A set is more efficient than a list for lookups, etc.
        with open(USERS_LIST_PATH, 'r') as users_list_file:
            list_of_lines = users_list_file.read().splitlines()
            target_users = set(list_of_lines)

        user_ids_with_unfetched_demographics = []

        # Read the demographics CSV file
        with open(DEMOGRAPHICS_CSV_PATH, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the first row (header)
            for row in csv_reader:
                user_id = row[3][5:]

                # If this user is not one of the target users, skip the rest of the loop.
                if user_id not in target_users:
                    continue

                index = int(row[0])

                # For some users (less than 1k out of the total 250k users in the ASI dataset) the demographics
                # are not fetched in the CSV file, and they contain 'NA' in all the demographics fields.
                # If this is the case we will skip the user. We check for an 'NA' in any of the age_gender/race fields,
                # even though they tend to be 'NA' all together.
                demographics_index_numbers = list(range(9, 21)) + list(range(26, 30))  # [9–20, 26–29]
                equals_na = []
                for i in demographics_index_numbers:
                    if row[i] == 'NA':
                        equals_na.append(True)
                    else:
                        equals_na.append(False)
                # If any of the above elements are equal to 'NA', skip the user
                if any(equals_na):
                    user_ids_with_unfetched_demographics.append(user_id)
                    continue


                gender_age = {'Female [0,25)':  float(row[9]),
                              'Female [25,34]': float(row[10]),
                              'Female [35,44]': float(row[11]),
                              'Female [45,54]': float(row[12]),
                              'Female [55,64]': float(row[13]),
                              'Female [65,..)': float(row[14]),
                              'Male [0,25)':  float(row[15]),
                              'Male [25,34]': float(row[16]),
                              'Male [35,44]': float(row[17]),
                              'Male [45,54]': float(row[18]),
                              'Male [55,64]': float(row[19]),
                              'Male [65,..)': float(row[20]),
                              }
                race = {'Asian': float(row[26]),
                        'Black': float(row[27]),
                        'Hispanic': float(row[28]),
                        'White': float(row[29]),
                        }
                location = row[31]  # The 'metro' attribute in ASI's dataset

                gender, age = decide_gender_age(gender_age)

                # Construct a *User* object
                this_user = User(user_id)
                # Add this user to the dataset
                self.add_user(this_user)
                # Set the attributes of the user
                # TODO: Race
                this_user.gender = gender
                this_user.age = age
                this_user.location = location

        logger.info('@ %.2f seconds: Finished loading the labels (demographics): %s users',
                    time.process_time(), format(len(self.users), ',d'))
        logger.info('%s users were skipped due to unfetched demographics: %s',
                    format(len(user_ids_with_unfetched_demographics), ',d'), user_ids_with_unfetched_demographics)

    def load_tweets(self, CSVS_ROOT_DIR):
        """Load all tweets from CSV files onto the User objects.

        Run this method after the *load_labels_and_create_users()* method, as the User objects are expected to
        already exist.

        Args:
            CSVS_ROOT_DIR: The root directory containing CSV files of users (all tweets of user)
        """

        logger.info('Loading the tweets of the ASI dataset from: "%s"', CSVS_ROOT_DIR)

        num_processed_users = 0  # Initialize the counter

        # Go through every file in the directory and its subdirectories,
        # and load the tweets into Tweet objects.
        for root, dirs, files in os.walk(CSVS_ROOT_DIR):
            for filename in files:
                if fnmatch.fnmatch(filename, '*.csv'):
                    csv_file_path = os.path.join(root, filename)
                    user_id = filename[:-4]
                    # Get the existing corresponding user object
                    this_user = self.get_user(user_id)
                    # If the user ID did not match any of the existing users, warn the user and continue to the next
                    # user
                    if this_user is None:
                        logger.warning('User ID "%s" did not match that of any existing users.'
                                       'Skipped loading the tweets for this user.', user_id)
                        continue
                    # Counter++
                    num_processed_users += 1
                    # Go through the tweets in the CSV file
                    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        next(csv_reader)  # Skip the first row (header)
                        for row in csv_reader:
                            # Create a *Tweet* object
                            # The initializer of the Tweet class will also call the *add_tweet* method for the user
                            Tweet(this_user, row[0], row[1], row[2], row[3])
                    # Now that all tweets of this user have been added to it, sort the tweets by date and time
                    this_user.sort_tweets_by_datetime()

                    # Log the progress
                    if num_processed_users % 1000 == 0:
                        logger.info('@ %.2f seconds: Finished loading the tweets of user #%s',
                                    time.process_time(), format(num_processed_users, ',d'))

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

    def drop_all_retweets(self, drop=True):
        """Remove any tweets of the dataset that are retweets.

        Calls the *User.drop_retweets()* method for all the users in the dataset.

        Args:
            drop: Boolean. When False, the method doubles as a counter without removing any of the retweets.

        Returns:
            A NumPy array containing the count of retweets for each user in the dataset
        """

        retweet_counts = []

        for user in self.users:
            retweet_count = user.drop_retweets(drop=drop)
            retweet_counts.append(retweet_count)

        return np.array(retweet_counts)

    def drop_all_short_tweets(self, min_word_count, drop=True):
        """Remove any tweets of the dataset with fewer words than the given threshold

        Calls the *User.drop_short_tweets()* method for all the users in the dataset.

        Args:
            min_word_count (int): The minimum acceptable word count. Any tweet with fewer words will be removed.
            drop (boolean): When False, the method doubles as a counter without removing any of the short tweets.

        Returns:
            A NumPy array containing the count of dropped tweets for each user in the dataset.
        """

        dropped_tweets_counts = []

        for user in self.users:
            dropped_tweets_counts.append(user.drop_short_tweets(drop=drop))

        return np.array(dropped_tweets_counts)

    def drop_all_foreign_tweets(self):
        """TODO"""

    def drop_users_with_no_tweets(self):
        """Remove any users of the dataset that have no tweets.

        Returns:
            A list of user IDs of the dropped users
        """

        dropped_user_ids = []
        users_to_remove = []

        for user in self.users:
            if user.get_num_tweets() == 0:
                # Refer to the *User.drop_retweets()* method (same idea).
                users_to_remove.append(user)

        for user in users_to_remove:
            dropped_user_ids.append(user.get_id())
            self.remove_user(user)

        return dropped_user_ids

    def get_user(self, user_id):
        """Find a user based on user ID

        The method returns the first User object from the dataset which has the given user ID.
        If the user ID does not match any that of the User objects, the method will return *None*.

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

    def produce_stats(self):
        """Produce statistics of the dataset and write it to a CSV file."""

        CSV_FILENAME = RUN_TIMESTAMP + ' ' + 'ASI dataset stats.csv'
        CSV_PATH = os.path.join('data/out', CSV_FILENAME)

        # Header of the CSV file and rows to write
        header = ['user_id', 'gender', 'age', 'num_tweets', 'less_than_3_words',
                  'lang_other', 'lang_na', 'lang_en', 'lang_und',
                  'total_num_words', 'newest_datetime', 'oldest_datetime']
        rows = []

        for user in self.users:
            user_id = user.get_id()
            gender = user.gender
            age = user.age
            num_tweets = user.get_num_tweets()
            total_num_words = np.sum(user.get_word_counts())  # total number of words in all tweets of the user
            datetimes = user.get_datetimes()
            if len(datetimes) != 0:
                # If the list is not empty
                newest_datetime = max(datetimes)
                oldest_datetime = min(datetimes)
            else:
                # If the list is empty (when the user has no tweets)
                newest_datetime = 'NA'
                oldest_datetime = 'NA'

            less_than_3_words = user.drop_short_tweets(min_word_count=3, drop=False)

            # Get the statistics of language of tweets for the user
            num_lang_en, num_lang_na, num_lang_und, num_lang_other = user.drop_foreign_tweets(drop=False)

            row = [user_id, gender, age, num_tweets, less_than_3_words,
                   num_lang_other, num_lang_na, num_lang_en, num_lang_und,
                   total_num_words, newest_datetime, oldest_datetime]
            rows.append(row)

        # Create the directory if it does not exist.
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

        # Write to the CSV file
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csv_output_file:
            csv_writer = csv.writer(csv_output_file)
            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        logger.info('@ %.2f seconds: Finished writing the dataset stats to CSV file: %s', time.process_time(), CSV_PATH)


class User:
    """Twitter user class"""

    def __init__(self, user_id):  # Initializer (instance attributes)
        self.__id = user_id  # Private data field
        self.__tweets = []

        # Attributes loaded by the *Dataset.load_labels_and_create_users()* method:
        self.gender = None
        self.age = None
        self.race = None
        self.location = None

    def add_tweet(self, tweet):  # Instance method
        """Add a tweet object to the user"""
        self.__tweets.append(tweet)

    def remove_tweet(self, tweet):
        """Remove a tweet object from the user's tweets"""

        self.__tweets.remove(tweet)
        # ↳ *array.remove(x)* removes the first occurrence of x from the array.

    def sort_tweets_by_datetime(self):
        """Sort the list of tweets of the user by date and time, in ascending order.

        Run this function once for each user object, after you loaded all tweets of the user.
        """

        self.__tweets.sort(key=lambda x: x.datetime)

    def drop_retweets(self, drop=True):
        """Remove any tweets of the user that are retweets.

        Args:
            drop (boolean): When False, the method doubles as a counter without removing any of the retweets.

        Returns:
            The number of retweets of the user
        """

        tweets_to_remove = []

        for tweet in self.__tweets:
            if tweet.is_retweet():
                # Removing the item from the list would shift all the subsequent items up, causing the loop to miss
                # the item directly after the removed item. To overcome this issue, we make a list of the items to
                # remove, and remove them altogether after this loop.
                tweets_to_remove.append(tweet)

        retweet_count = len(tweets_to_remove)

        # Remove the tweets from the user
        if drop:
            for tweet in tweets_to_remove:
                self.remove_tweet(tweet)

        return retweet_count

    def drop_short_tweets(self, min_word_count, drop=True):
        """Remove any tweets of the user with fewer words than the given threshold.

        Args:
            min_word_count (int): The minimum acceptable word count. Any tweet with fewer words will be removed.
            drop (boolean): When False, the method doubles as a counter without removing any of the short tweets.

        Returns:
            (int) Number of dropped tweets.
        """

        tweets_to_remove = []

        for tweet in self.__tweets:
            if tweet.word_count < min_word_count:
                # Refer to the *User.drop_retweets()* method (same idea).
                tweets_to_remove.append(tweet)

        if drop:
            for tweet in tweets_to_remove:
                self.remove_tweet(tweet)

        return len(tweets_to_remove)

    def drop_foreign_tweets(self, drop=True):
        """Remove non-English tweets of the user.

        In the ASI dataset, a tweet's language can be:
            - 'en':  English
            - 'und': Undetermined. Usually a tweet with no words (only mentions and URLs and emoticons)
            - 'NA':  These are tweets that for some reason are not updated with the language, so we don't know.
            - Other (e.g., 'fr', 'es'): Foreign tweets—tweets in other languages (e.g., French, Spanish)

        We drop all the 'und' tweets and keep all the 'en' tweets.
        We will also drop all the foreign tweets (tweets with other languages).

        We decide to drop or keep the 'NA' tweets based on the following proportion: en/(en+foreign)
        Where:
            en      = number of English tweets
            foreign = number of foreign tweets—tweets in other languages
        The 'und' and 'NA' tweets are left out of the above numbers.
        - If the proportion of English/(English+Foreign) tweets is >= 95%, we keep all the 'NA' tweets as there is a
        good chance that they are in English. Otherwise, we drop all the 'NA' tweets to prevent introducing noise
        into the dataset.
        - TODO: For some users both the above numbers are zero (all tweets of the user are of language 'NA' or 'und').
        In those cases all of the tweets of the user will be dropped and we will end up losing that user.
        We will report the ocunt of those users.

        Args:
            drop (boolean): When False, the method doubles as a counter without removing any of the foreign tweets.

        Returns:
            (int) Number of dropped tweets.
        """

        num_en = 0
        num_na = 0
        num_und = 0
        num_foreign = 0

        tweets_to_remove = []

        for tweet in self.__tweets:
            if tweet.language == 'en':
                num_en += 1
            elif tweet.language == 'na':
                num_na += 1
            elif tweet.language == 'und':
                num_und += 1
                # Refer to the *User.drop_retweets()* method (same idea).
                tweets_to_remove.append(tweet)
            else:
                num_foreign += 1
                tweets_to_remove.append(tweet)

        # If *drop* == False, only return the statistics and skip the rest of the method
        if not drop:
            return num_en, num_na, num_und, num_foreign

        # • Decide to drop the 'NA' tweets or not
        if (num_en + num_foreign) == 0:
            # This would prevent a ZeroDivisionError
            drop_na_tweets = True
        else:
            en_proportion = num_en / (num_en + num_foreign)
            if en_proportion >= 0.95:
                drop_na_tweets = False
            else:
                drop_na_tweets = True

        # TODO: Return something! + Is this the way to go with NA tweets?


    def get_id(self):
        """Getter (accessor) for user ID"""
        return self.__id

    def get_tweets(self):
        """Getter (accessor) for the user's tweets"""
        return self.__tweets

    def get_num_tweets(self):
        """Get number of tweets of the user"""
        return len(self.__tweets)

    def get_word_counts(self):
        """Get the word counts of the tweets of the user

        Returns:
             A NumPy array containing the word count for each tweet of the user
        """

        word_counts = []
        for tweet in self.__tweets:
            word_counts.append(tweet.word_count)

        return np.array(word_counts)

    def get_datetimes(self):
        """Get the date and time of the tweets of the user

        Returns:
            A list containing a *datetime* object for each tweet of the user
        """

        datetimes = []
        for tweet in self.__tweets:
            datetimes.append(tweet.datetime)

        return datetimes


class Tweet:
    """Tweet class"""

    def __init__(self, user, datetime_string, language, original_text, url):
        # Remove the colon from the UTC offset (last six characters): '+00:00'
        datetime_string = datetime_string[:-5] + datetime_string[-5:-3] + datetime_string[-2:]
        # Parse the string into a *datetime* object
        self.datetime = datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S%z')

        self.__user = user
        user.add_tweet(self)

        self.language = language
        self.original_text = original_text
        self.text = None  # Will be assigned by the *preprocess()* instance method
        self.url = url
        self.__retweet = None  # Will be assigned by the *preprocess()* instance method
        self.word_count = None  # Will be assigned by the *preprocess()* instance method

    def get_user(self):
        """Getter (accessor) for the user attribute"""
        return self.__user

    def is_retweet(self):
        """Getter (accessor) for the boolean *retweet* attribute"""
        return self.__retweet

    def preprocess(self):
        """Preprocess the tweet (only if it is not a retweet)

        - Determine whether the tweet is a retweet or not, and update its boolean *retweet* attribute accordingly.
        - If not a retweet, preprocess the tweet and update its *text* attribute with the processed text. Also, update
            its *word_count* attribute.
        """

        # Determine whether the tweet is a retweet or not
        if self.original_text[:3] == 'RT ':
            self.__retweet = True
        else:
            self.__retweet = False
            # Preprocess the tweet
            self.text, ignored1, self.word_count, ignored2, ignored3 = preprocess_tweet(self.original_text,
                                                                                        output_mode='multiple')


def decide_gender_age(gender_age):
    """Decide the gender based on the gender-age demographics

    Note that these users have already been filtered based on their demographics, and sum of the gender probabilities
    are expected to be >=0.98 OR <=0.02. Hence, here we judge only based on a simple comparison of the two sums.

    Also for the age groups, we expect a similar condition in the selected users. Here, we only select the age group
    with the maximum probability (summed over the two genders).

    Args:
        gender_age: A dictionary of gender and age demographics with the following 12 keys and float values:
            'Female [0,25)'
            'Female [25,34]'
            'Female [35,44]'
            'Female [45,54]'
            'Female [55,64]'
            'Female [65,..)'
            'Male [0,25)'
            ...

    Returns:
        gender <string>: One of the two labels: 'female' or 'male'
        age <string>: One of the six labels: '[0, 25]', ..., '[65,..)'
    """

    ## Decide the gender label...
    KEYS_FEMALE = ['Female [0,25)', 'Female [25,34]', 'Female [35,44]', 'Female [45,54]', 'Female [55,64]', 'Female [65,..)']
    KEYS_MALE = ['Male [0,25)', 'Male [25,34]', 'Male [35,44]', 'Male [45,54]', 'Male [55,64]', 'Male [65,..)']

    sum_female = 0
    for key in KEYS_FEMALE:
        sum_female += gender_age[key]

    sum_male = 0
    for key in KEYS_MALE:
        sum_male += gender_age[key]

    if sum_female > sum_male:
        gender = 'female'
    else:
        gender = 'male'


    ## Decide the age label...
    AGES = ['[0,25)', '[25,34]', '[35,44]', '[45,54]', '[55,64]', '[65,..)']
    GENDERS = ['Female', 'Male']
    sum_ages = []

    # Sum the age probabilities for each age group over the two genders
    for age in AGES:
        sum_ages.append(gender_age['Female ' + age] + gender_age['Male ' + age])

    # • Find the age group with the maximum probability (summed over two genders)
    # We will sort both lists based on *sum_ages* in descending order.
    sorted_sum_ages, sorted_ages = zip(*sorted(zip(sum_ages, AGES), reverse=True))
    # The first item in the sorted lists is the one with the maximum sum probability
    age = sorted_ages[0]

    return gender, age


def main():
    """The main function"""

    # Demographics (labels)
    DEMOGRAPHICS_CSV_PATH = 'data/Advanced Symbolics/2018-10-30 user_info, Canada Ignite 5.csv'
    # USERS_LIST_PATH = 'data/Advanced Symbolics/List of users _ Batch 1 (934).txt'
    USERS_LIST_PATH = 'data/Advanced Symbolics/List of users _ Batch 1–7 (30,934).txt'

    # Tweets
    # CSV_BATCHES_DIR = 'P:/2018-12-20_13-31-03 _ Batch 1'
    CSV_BATCHES_DIR = 'P:/'

    dataset = Dataset()  # Constructor
    # Load the labels (demographics) and create users
    dataset.load_labels_and_create_users(DEMOGRAPHICS_CSV_PATH, USERS_LIST_PATH)
    # Load the tweets onto the existing users
    dataset.load_tweets(CSV_BATCHES_DIR)

    dataset.preprocess_tweets()

    #  • Pickle the dataset object
    PICKLES_DIR = 'data/out/pickles'
    DATASET_PICKLE_FILENAME = RUN_TIMESTAMP + ' dataset object.pickle'
    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(PICKLES_DIR), exist_ok=True)
    # Pickle
    with open(os.path.join(PICKLES_DIR, DATASET_PICKLE_FILENAME), 'wb') as pickle_output_file:
        pickle.dump(dataset, pickle_output_file)
    logger.info('@ %.2f seconds: Finished pickling the dataset', time.process_time())

    # TEMP
    tweet_counts_before = dataset.get_num_tweets()

    # Drop all the retweets
    retweet_counts = dataset.drop_all_retweets()
    logger.info('@ %.2f seconds: Finished removing the retweets', time.process_time())
    logger.info('A total of %s retweets were dropped.', format(sum(retweet_counts), ',d'))

    # TEMP
    tweet_counts_after = dataset.get_num_tweets()
    if np.array_equal(tweet_counts_before - retweet_counts, tweet_counts_after):
        logger.info('(TEMP) Success: The counts of tweets and retweets before and after drop checks out!')

    # TODO
    # dataset.drop_all_foreign_tweets()
    # dataset.drop_all_short_tweets()

    # At this point, some users might have no tweets, either because an inconsistency between the list of users (when
    # loading the user demographics) and the existing tweets dataset, or because all of their tweets were retweets
    # or short tweets and were dropped in the previous operations.
    # Let's drop those users
    dropped_user_ids = dataset.drop_users_with_no_tweets()
    logger.info('Dropped %s users with no tweets: %s', format(len(dropped_user_ids), ',d'), dropped_user_ids)

    dataset.produce_stats()

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
    proj_dir = 'C:/Users/Saman/GitHub/usermodeling'
    logger, RUN_TIMESTAMP = my_utils.configure_root_logger(proj_dir)
    my_utils.set_working_directory(proj_dir)
    main()
