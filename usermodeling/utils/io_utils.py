"""Utilities related to disk I/O"""

import csv
import logging
import os


def write_iterable_to_csv(iterable, iterable_name, log_file_path):
    """Write an iterable to a CSV file.

    This function writes any iterable object to a CSV file next to the log file.
    - You can get *log_file_path* by calling *logger.handlers[1].baseFilename* in the root module, assuming that
    the file handler is the second handler of the logger.

    • CSV Writer objects remarks:
    - *csvwriter.writerow(row)*:   A row must be an iterable of strings or numbers.
    - *csvwriter.writerows(rows)*: *rows* must be a list of row objects, described above.
    """

    # Determine the path of the output CSV file based on the path of the log file, such that the leading date and
    # time of the two filenames are the same.
    log_file_directory = os.path.dirname(log_file_path)
    log_file_name_without_extension = os.path.splitext(os.path.basename(log_file_path))[0]
    CSV_PATH = os.path.join(log_file_directory, log_file_name_without_extension + "; " + iterable_name + ".csv")

    # • Find out if the iterable is an “iterable of iterables”. For example, [[1, 2], [3, 4]] is an iterable
    # of iterables—each item in it is also an iterable; however, [1, 2, 3] isn't.

    # Select the first item in the iterable. We will only test this item.
    item = iterable[0]

    # The following is “the try statement”.
    try:
        iterator = iter(item)
        # ↳ This will raise a TypeError exception if *item* is not iterable.
    except TypeError:
        # This means *item* is not iterable.
        item_is_iterable = False
    else:
        # This means *item* is an iterable.
        item_is_iterable = True

    # If *item* is a string, it means it escaped from us! Strings are considered iterables, but here, we are
    # looking for iterables such as lists and tuples, not strings.

    # If *item* is not iterable or it is a string, convert *iterable* to a list of lists of one item each.
    # For example: (1, 2, 3) → [[1], [2], [3]]
    if not item_is_iterable or isinstance(item, str):
        iterable = [[item] for item in iterable]
    # Now *iterable* is an “iterable of iterables”!

    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write to the CSV file
    with open(CSV_PATH, 'w', newline='', encoding="utf-8") as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        csv_writer.writerow([iterable_name])
        csv_writer.writerows(iterable)

    logger.info('%s was written to CSV file: "%s"', iterable_name, CSV_PATH)


def write_feature_importance_rankings_to_csv(sorted_feature_weights, sorted_feature_names, log_file_path):
    """Write the feature importance rankings to a CSV file.

    This function writes the feature importance rankings to a CSV file, next to the log file.
    Refer to the docstring of the *write_iterable_to_csv()* function.
    """

    # Determine the path of the output CSV file based on the path of the log file, such that the leading date and
    # time of the two filenames are the same.
    log_file_directory = os.path.dirname(log_file_path)
    log_file_name_without_extension = os.path.splitext(os.path.basename(log_file_path))[0]
    CSV_PATH = os.path.join(log_file_directory, log_file_name_without_extension + "; Feature importance rankings.csv")

    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write to the CSV file
    with open(CSV_PATH, 'w', newline='', encoding="utf-8") as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        csv_writer.writerow(["Feature weights:", "Feature names:"])
        csv_writer.writerows(zip(sorted_feature_weights, sorted_feature_names))

    logger.info('List of features based on their importance in the classification model (absolute feature weight) '
                'was written to CSV file: "%s"', CSV_PATH)


'''
The following lines will be executed any time this .py file is run as a script or imported as a module.
'''
# Create a logger object. The root logger would be the parent of this logger
# Note that if you run this .py file as a script, this logger will not function, because it is not configured.
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # The following lines will be executed only if this .py file is run as a script,
    # and not if it is imported as a module.
    print("Module was executed directly.")
