'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Jiahao (Derek) Ye
CS 251 Data Analysis Visualization, Spring 2020
'''
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Update the counts of each words in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    - When reading in email files, you might experience errors due to reading funky characters
    (spam can contain weird things!). On this dataset, this can be fixed by telling Python to assume
    each file is encoded using 'latin-1': encoding='latin-1'
    '''

    all_files = os.listdir(email_path)  
    listOfFiles = []
    counts = {}

    for (dirpath, dirnames, filenames) in os.walk(email_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.txt']

    for i in range(len(listOfFiles)):
        with open(listOfFiles[i], encoding='latin-1') as file:
            content = file.read()
            listOfWords = tokenize_words(content)
            for i in listOfWords:
                if i in counts:
                    counts[i] += 1
                else:
                    counts[i] = 1
        file.close()

    return counts, len(listOfFiles)
   

    pass


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''

    top_words = np.array(sorted(word_freq.items(),key=lambda item:item[1], reverse=True))
    words = top_words[:num_features, 0].tolist()
    counts = top_words[:num_features, 1].tolist()

    return words, counts

    pass


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    
    all_files = os.listdir(email_path)  
    listOfFiles = []
    classes = np.zeros((num_emails))
    counts = np.zeros((num_emails,len(top_words)))

    for (dirpath, dirnames, filenames) in os.walk(email_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.txt']
                
    for i in range(len(listOfFiles)):
        if listOfFiles[i][11:15] != "spam":
            classes[i] = 1
        features = {}
        with open(listOfFiles[i], encoding='latin-1') as file:
            content = file.read()
            listOfWords = tokenize_words(content)
            for j in top_words:
                features[j] = listOfWords.count(j)
        file.close()
        counts[i,:] = np.array(list(features.values()))
    
    return counts, np.array(classes)
                
    pass


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].

    HINTS:
    - If you're shuffling, work with indices rather than actual values.
    '''
    
    N = features.shape[0]

    if shuffle == True:
        inds = np.arange(N)
        np.random.shuffle(inds)
        temp_feature = features.copy()
        temp_y = y.copy()
        for i in range(N):
            temp_feature[i] = features[inds[i]]
            temp_y[i] = y[inds[i]]
        features = temp_feature
        y = temp_y
    else:
        inds = np.arange(N)

    idx_bound_test = int(features.shape[0] * (1-test_prop))

    x_split = np.split(features, [idx_bound_test + 1, N], axis = 0)
    y_split = np.split(y, [idx_bound_test + 1, N], axis = 0)
    idx_split = np.split(np.array(inds), [idx_bound_test + 1, N])

    x_train = x_split[0]
    y_train = y_split[0]
    inds_train = idx_split[0]
    x_test = x_split[1]
    y_test = y_split[1]
    inds_test = idx_split[1]

    return x_train, y_train, inds_train, x_test, y_test, inds_test


    pass

def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''

    all_files = os.listdir(email_path)  
    listOfFiles = []

    for (dirpath, dirnames, filenames) in os.walk(email_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.txt']

    allcontent = []
    
    for i in inds:
        with open(listOfFiles[i], encoding='latin-1') as file:
            content = file.read()
            allcontent.append(content)
        file.close()
    
    return allcontent

    pass
