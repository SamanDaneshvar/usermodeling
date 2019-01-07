""" This module loads the dataset of the Author Profiling task at PAN 2018 and pre-processes it.
    For more information refer to the docstring of the *loadData* function.

    Remarks:
    - The *xml.etree.ElementTree* module is not secure against maliciously constructed data. Make sure the XML files
    are from a trusted source.

    Author: Saman Daneshvar
"""

import logging
import time
import os
import fnmatch
import sys
import shutil
import csv

from xml.etree import ElementTree


def loadPanData(xmlsDirectory, truthPath, writeToTxtFiles=False, txtsDestinationDirectory=None):
    """ This function loads the PAN dataset and the truth, parses the XML and returns:
        Merged tweets of the authors, the truth, Author IDs, and the original length of the tweets.
        It also writes the tweets to TXT files (optional).

        Remarks:
        - Since *xmlFilenames* is sorted in ascending order, all the returned lists will also be in the same order
        (sorted in ascending order of the Author IDs).
        - List of replacements:
            Line feed		<LineFeed>
            End of Tweet	<EndOfTweet>
    """

    ''' *os.listdir* returns a list containing the name of all files and folders in the given directory.
        Normally, the list is created in ascending order. However, the Python documentation states,
        “the list is in arbitrary order”.
        To ensure consistency and avoid errors in syncing the order of the items among
        different lists (e.g., *authorIDs*, *truths*), we sort the list by calling *sorted*.
        *sorted()* returns a new sorted list (in ascending lexicographical order) of all the items in an iterable.
    '''
    xmlFilenames = sorted(os.listdir(xmlsDirectory))

    # Store the Author IDs in a list
    # The Author IDs list will have the same order as the XML filenames list.
    authorIDs = []  # Create an empty list
    for xmlFilename in xmlFilenames:
        authorIDs.append(xmlFilename[:-4])

    # Skip loading truth if path input is None. Else, load the truth from the file.
    if truthPath is None:
        logger.info("*truthPath* is None => Skipped loading the truth")
        truths = None
        # This scenario will happen when loading the test dataset for **TIRA** evaluation, where the truth of the test
        # set is not provided.
    else:
        truths = loadTruth(truthPath, authorIDs)

    if writeToTxtFiles:
        logger.info("The parsed XMLs will also be written to TXT files.")
        # Create the directory if it does not exist.
        os.makedirs(txtsDestinationDirectory, exist_ok=True)

    # Initialize the lists.
    # The lists will have the same order as the XML filenames list (refer to: “Iterate over XML Files”)
    originalTweetLengths = []  # Create an empty list
    # ↳ Every row will represent an author, every column will represent a tweet.
    mergedTweetsOfAuthors = []  # Create an empty list
    # ↳ Each cell will contain all 100 tweets of an author, merged.

    # Iterate over XML files
    for authorIndex, xmlFilename in enumerate(xmlFilenames):
        # Make sure only XML files go through
        if not fnmatch.fnmatch(xmlFilename, '*.xml'):
            logger.error("Encountered a non-XML file inside the directory: %s >>> The program will exit now.",
                         xmlFilename)
            # Exit the program
            sys.exit()

        # Read the XML file and parse it into a tree
        # Parser is explicitly defined to ensure UTF-8 encoding.
        tree = ElementTree.parse(os.path.join(xmlsDirectory, xmlFilename),
                                 parser=ElementTree.XMLParser(encoding="utf-8"))
        root = tree.getroot()
        ''' root is the root element of the parsed tree
            root[0], ..., root[m-1] are the children of root—elements one level below the root.
            root[0][0], ..., root[0][n-1] are the children of root[0].
            and so on.
            
            Each element has a tag, a dictionary of attributes, and sometimes some text:
                root[i][j].tag, ”.attrib, ”.text 
        '''

        # Add an empty new row to the list. Each row represents an author.
        originalTweetLengths.append([])

        # Initialize the list. Note that this list resets in every author (XML file) loop.
        tweetsOfThisAuthor = []  # Create an empty list

        # Iterate over the tweets within this parsed XML file:
        # Record the tweet length, replace line feeds, and append the tweet to a list
        for child in root[0]:
            # Element.text accesses the element's text content,
            # which is saved with the following format in the XML files: <![CDATA[some text]]>
            tweet = child.text
            originalTweetLengths[authorIndex].append(len(tweet))

            # Replace line feed (LF = \n) with “ <LineFeed> ”
            # Note: There were no carriage return (CR = \r) characters in any of the 3,000 XML files.
            tweet = tweet.replace('\n', " <LineFeed> ")

            # Create a list of the tweets of this author, to write to a text file and merge, after the loop terminates.
            ''' Google Python Style Guide: Avoid using the + and += operators to accumulate a string within a loop.
                Since strings are immutable, this creates unnecessary temporary objects and results in quadratic rather
                than linear running time.
                Avoid: mergedTweetsOfAuthors[authorIndex] += tweet + " <EndOfTweet> "
                Instead, append each substring to a list and ''.join the list after the loop terminates.
            '''
            tweetsOfThisAuthor.append(tweet)

        # Write the tweets of this author to a TXT file
        # Note that in these tweets, the line feed characters are replaced with a tag.
        if writeToTxtFiles:
            # Create a TXT file with the Author ID as the filename (same as the XML files) in the write mode
            with open(os.path.join(txtsDestinationDirectory, authorIDs[authorIndex] + ".txt"),
                      'w', encoding="utf-8") as txtOutputFile:
                txtOutputFile.write('\n'.join(tweetsOfThisAuthor))
                # ↳ '\n'.join adds a newline character between every two strings,
                # so there won't be any extra line feeds on the last line of the file.

        # Concatenate the tweets of this author, and append it to the main list
        mergedTweetsOfThisAuthor = " <EndOfTweet> ".join(tweetsOfThisAuthor) + " <EndOfTweet>"
        # ↳ " <EndOfTweet> ".join adds the tag between every two strings, so we need to add another tag to the end.
        mergedTweetsOfAuthors.append(mergedTweetsOfThisAuthor)

    logger.info("@ %.2f seconds: Finished loading the dataset", time.process_time())

    return mergedTweetsOfAuthors, truths, authorIDs, originalTweetLengths


def loadTruth(truthPath, authorIDs):
    """ This function loads the truth from the TXT file,
        and makes sure the order of the Truth list is the same as the Author IDs list.
    """

    # Load the Truth file, sort its lines (in ascending order), and store them in a list
    tempSortedAuthorIDsAndTruths = []  # Create an empty list
    # ↳ Each row represents an author. Column 0: Author ID,  column 1: Truth.
    with open(truthPath, 'r') as truthFile:
        for line in sorted(truthFile):
            # ↳ “for line” automatically skips the last line if it only contains a newline character.
            # ↳ *sorted()* returns a new sorted list (in ascending lexicographical order) of all the items in an
            # iterable—here, the lines in truthFile.
            # Remove the ending newline character from each line (line is a string)
            line = line.rstrip('\n')
            # str.split returns a list of the parts of the string which are separated by the specified separator string.
            tempSortedAuthorIDsAndTruths.append(line.split(":::"))

    truths = []  # Create an empty list
    # Make sure the rows in *tempSortedAuthorIDsAndTruths* and *authorIDs* have the same order,
    # and store the truth in the *truths* list.
    for i, row in enumerate(tempSortedAuthorIDsAndTruths):
        # Compare the Author ID in the two lists
        if row[0] == authorIDs[i]:
            # ↳ row[0] is the Author ID of this row, and row[1] is the truth of this row.
            # Add the truth to the truths list
            truths.append(row[1])
        else:
            logger.error("Failed to sync the order of the Truth list and the Author ID list."
                         "Row number: %d >>> The program will now exit.", i)
            # Exit the program
            sys.exit()

    return truths


def splitTrainAndTestFiles(authorIDs_train, authorIDs_test, truths_train, truths_test, presetKey):
    """ This function splits the XML files of the dataset into training and test sets according to the results of
    sklearn's *train_test_split* function. It also writes two separate TXT files for the truth of the training
    and test sets. This function is used for mimicking the **TIRA** environment for local testing.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'datasetName': 'PAN 2018 English',
                                            'xmlsSourceDirectory': 'data/en/text/',
                                            'xmlsDestinationSubdirectory': 'en/text/',
                                            'truthDestinationSubpath': 'en/truth.txt',
                                            },
                          'PAN18_Spanish': {'datasetName': 'PAN 2018 Spanish',
                                            'xmlsSourceDirectory': 'data/es/text/',
                                            'xmlsDestinationSubdirectory': 'es/text/',
                                            'truthDestinationSubpath': 'es/truth.txt',
                                            },
                          'PAN18_Arabic': {'datasetName': 'PAN 2018 Arabic',
                                            'xmlsSourceDirectory': 'data/ar/text/',
                                            'xmlsDestinationSubdirectory': 'ar/text/',
                                            'truthDestinationSubpath': 'ar/truth.txt',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[presetKey]

    # Define the constants: Destination main directory of the training and test datasets
    TRAIN_DESTINATION_MAIN_DIR = "data/TiraDummy/trainDirectory/"
    TEST_DESTINATION_MAIN_DIR = "data/TiraDummy/testDirectory/"

    # Assemble the destination directories and paths
    xmlsDestinationDirectory_train = os.path.join(TRAIN_DESTINATION_MAIN_DIR, PRESET['xmlsDestinationSubdirectory'])
    xmlsDestinationDirectory_test = os.path.join(TEST_DESTINATION_MAIN_DIR, PRESET['xmlsDestinationSubdirectory'])
    truthDestinationPath_train = os.path.join(TRAIN_DESTINATION_MAIN_DIR, PRESET['truthDestinationSubpath'])
    truthDestinationPath_test = os.path.join(TEST_DESTINATION_MAIN_DIR, PRESET['truthDestinationSubpath'])

    # Create the destination directories if they do not exist.
    for directory in [xmlsDestinationDirectory_train, xmlsDestinationDirectory_test,
                      os.path.dirname(truthDestinationPath_train), os.path.dirname(truthDestinationPath_test)]:
        os.makedirs(directory, exist_ok=True)

    # Copy the XML files of the split training and test dataset to two different destinations.
    for authorID in authorIDs_train:
        shutil.copy(os.path.join(PRESET['xmlsSourceDirectory'], authorID + ".xml"), xmlsDestinationDirectory_train)
    for authorID in authorIDs_test:
        shutil.copy(os.path.join(PRESET['xmlsSourceDirectory'], authorID + ".xml"), xmlsDestinationDirectory_test)

    # • Write the truth of the split training and test datasets to two different text files.
    # Create empty lists
    linesToWrite_train = []
    linesToWrite_test = []
    # Iterate over the authors in the training and test set, and keep the lines to be written
    for authorID, gender in zip(authorIDs_train, truths_train):
        linesToWrite_train.append(authorID + ":::" + gender)
    for authorID, gender in zip(authorIDs_test, truths_test):
        linesToWrite_test.append(authorID + ":::" + gender)
    # Write the lines to the files
    with open(truthDestinationPath_train, 'w') as truthFile_train:
        truthFile_train.write('\n'.join(linesToWrite_train))
    with open(truthDestinationPath_test, 'w') as truthFile_test:
        truthFile_test.write('\n'.join(linesToWrite_test))
        # ↳ '\n'.join adds a newline character between every two strings,
        # so there won't be any extra line feeds on the last line of the file.

    logger.info("@ %.2f seconds: Finished splitting the files of the training and test datasets (to mimic TIRA)",
                time.process_time())


def loadFlameDictionary(path="data/Flame_Dictionary.txt"):
    """ This function reads the Flame Dictionary from a text file and returns:
        1. *flameDictionary*: A Python dictionary with all the entries
            Keys:   (string) Expression
            Values: (int)    Flame level
        2. *flameExpressionsDict*: A Python dictionary with the entries, separated by Flame level, into five lists
            Keys:   (int)           Flame level
            Values: (list: strings) Expressions

        Remarks:
        - If there are any duplicate expressions in the text file, the value of the first instance is kept, and the
        other instances are reported as duplicates.
    """

    logger.info("Loading the Flame Dictionary from path: %s", os.path.realpath(path))

    flameDictionary = {}  # Create an empty dictionary
    duplicates = []  # Create an empty list
    flameExpressionsDict = {1: [], 2: [], 3: [], 4: [], 5: []}  # Create a dictionary of 5 empty lists

    with open(path, 'r') as flameDictionaryFile:
        for line in flameDictionaryFile:
            # ↳ “for line” automatically skips the last line if it only contains a newline character.
            # Remove the ending newline character from each line (line is a string)
            line = line.rstrip('\n')

            # Split the line into the flame level and the expression
            flameLevel = int(line[0])
            # ↳ int() converts the string into an integer.
            expression = line[2:]

            # Add the entry to the dictionary, and to the corresponding list within *flameExpressionsDict*.
            # If it already exists in the dictionary, ignore it and keep a record of it in the *duplicates* list.
            if expression in flameDictionary:
                duplicates.append(expression)
            else:
                flameDictionary[expression] = flameLevel
                flameExpressionsDict[flameLevel].append(expression)

    # Report the duplicate items to the user
    if len(duplicates) > 0:
        logger.warning("%d duplicate expressions found in the Flame Dictionary: %s",
                       len(duplicates), duplicates)

    return flameDictionary, flameExpressionsDict


def writePredictionsToXmls(authorIDs_test, y_predicted, xmlsDestinationMainDirectory, languageCode):
    """ This function is used only in **TIRA** evaluation.
        It writes the predicted results to XML files with the following format:
        <author id="author-id" lang="en|es" gender_txt="female|male" gender_img="N/A" gender_comb="N/A" />
    """

    # Add the alpha-2 language code (“en” or “es”) subdirectory to the end of the output directory
    xmlsDestinationDirectory = os.path.join(xmlsDestinationMainDirectory, languageCode)

    # Create the directory if it does not exist.
    os.makedirs(xmlsDestinationDirectory, exist_ok=True)

    # Iterate over authors in the test set
    for authorID, predictedGender in zip(authorIDs_test, y_predicted):
        # Create an *Element* object with the desired attributes
        root = ElementTree.Element('author', attrib={'id': authorID,
                                                     'lang': languageCode,
                                                     'gender_txt': predictedGender,
                                                     'gender_img': "N/A",
                                                     'gender_comb': "N/A",
                                                     })
        # Create an ElementTree object
        tree = ElementTree.ElementTree(root)
        # Write the tree to an XML file
        tree.write(os.path.join(xmlsDestinationDirectory, authorID + ".xml"))
        # ↳ ElementTree sorts the dictionary of attributes by name before writing the tree to file.
        # ↳ The final file would look like this:
        # <author gender_comb="N/A" gender_img="N/A" gender_txt="female|male" id="author-id" lang="en|es" />
    logger.info("@ %.2f seconds: Finished writing the predictions to XML files", time.process_time())


def writeFeatureImportanceRankingsToCSV(sortedFeatureWeights, sortedFeatureNames, logFilePath):
    """ This function writes the feature importance rankings to a CSV file, next to the log file.
    - Refer to the docstrings of the writeIterableToCSV() function.
    """

    # Determine the path of the output CSV file based on the path of the log file, such that the leading date and
    # time of the two filenames are the same.
    logFileDirectory = os.path.dirname(logFilePath)
    logFileNameWithoutExtension = os.path.splitext(os.path.basename(logFilePath))[0]
    CSV_PATH = os.path.join(logFileDirectory, logFileNameWithoutExtension + "; Feature importance rankings.csv")

    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write to the CSV file
    with open(CSV_PATH, 'w', newline='', encoding="utf-8") as csvOutputFile:
        csvWriter = csv.writer(csvOutputFile)
        csvWriter.writerow(["Feature weights:", "Feature names:"])
        csvWriter.writerows(zip(sortedFeatureWeights, sortedFeatureNames))

    logger.info('List of features based on their importance in the classification model (absolute feature weight) '
                'was written to CSV file: "%s"', CSV_PATH)


def writeIterableToCSV(iterable, iterableName, logFilePath):
    """ This function writes any iterable object to a CSV file next to the log file.
    - You can get *logFilePath* by calling *logger.handlers[1].baseFilename* in the root module, assuming that
    the file handler is the second handler of the logger.
    • CSV Writer objects remarks:
    - *csvwriter.writerow(row)*:   A row must be an iterable of strings or numbers.
    - *csvwriter.writerows(rows)*: *rows* must be a list of row objects, described above.
    """

    # Determine the path of the output CSV file based on the path of the log file, such that the leading date and
    # time of the two filenames are the same.
    logFileDirectory = os.path.dirname(logFilePath)
    logFileNameWithoutExtension = os.path.splitext(os.path.basename(logFilePath))[0]
    CSV_PATH = os.path.join(logFileDirectory, logFileNameWithoutExtension + "; " + iterableName + ".csv")

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
        itemIsIterable = False
    else:
        # This means *item* is an iterable.
        itemIsIterable = True

    # If *item* is a string, it means it escaped from us! Strings are considered iterables, but here, we are
    # looking for iterables such as lists and tuples, not strings.

    # If *item* is not iterable or it is a string, convert *iterable* to a list of lists of one item each.
    # For example: (1, 2, 3) → [[1], [2], [3]]
    if not itemIsIterable or isinstance(item, str):
        iterable = [[item] for item in iterable]
    # Now *iterable* is an “iterable of iterables”!

    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write to the CSV file
    with open(CSV_PATH, 'w', newline='', encoding="utf-8") as csvOutputFile:
        csvWriter = csv.writer(csvOutputFile)
        csvWriter.writerow([iterableName])
        csvWriter.writerows(iterable)

    logger.info('%s was written to CSV file: "%s"', iterableName, CSV_PATH)


# - - - - - - -
# The following lines will be executed any time this .py file is run as a script or imported as a module.

# Create a logger object. The root logger would be the parent of this logger
# Note that if you run this .py file as a script, this logger will not function, because it is not configured.
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # The following lines will be executed only if this .py file is run as a script,
    # and not if it is imported as a module.
    print("Module was executed directly.")
