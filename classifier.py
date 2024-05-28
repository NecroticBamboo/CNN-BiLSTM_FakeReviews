'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from https://archive-beta.ics.uci.edu/ml/datasets/spambase
It contains pre-processed metrics, such as the frequency of certain
words and letters, from a collection of emails. A classification for
each one indicating 'spam' or 'not spam' is in the final column.
See the linked page for full details of the data set.

This script uses three classifiers to predict the class of an email
based on the metrics. These are not representative of modern spam
detection systems.
'''

# Remember to update the script for the new data when you change this URL
# URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
AmazonDatasetLink = "Datasets/amazon_reviews.txt"
YelpDatasetLink = "Datasets/review_data.csv"
RestaurantDatasetLink = "Datasets/restaurant_reviews_anonymized.csv"
HotelDatasetLink = "Datasets\op_spam_v1.4"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from operator import indexOf
from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
import csv
import string
import re
import os

import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow import ops
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''

    # If your data is in an Excel file, install 'xlrd' and use
    # pandas.read_excel instead of read_table
    #from pandas import read_excel
    #frame = read_excel(URL)

    # If your data is in a private Azure blob, install 'azure-storage' and use
    # BlockBlobService.get_blob_to_path() with read_table() or read_excel()
    #from azure.storage.blob import BlockBlobService
    #service = BlockBlobService(ACCOUNT_NAME, ACCOUNT_KEY)
    #service.get_blob_to_path(container_name, blob_name, 'my_data.csv')
    #frame = read_table('my_data.csv', ...

    frame = read_table(
        URL,
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',
        #compression='bz2',

        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        header=None,
        #header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame

def loadData(path, rawData, preprocessedData, separator):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter = separator)
        (labelIndex, textIndex) = getLinesIndexes(reader)
        for line in reader:
            (Text, Label) = parseReview(line, labelIndex, textIndex)
            rawData.append((Text,Label))
            preprocessedData.append((preProcessText(Text),Label))
            

def prepareHotelData(rawData, preprocessedData):
    for subdir, dirs, files in os.walk(HotelDatasetLink):
        for file in files:
            f=open(os.path.join(subdir,file),'r')
            text=f.read()
            f.close()
            
            labelDirName = subdir.split('\\')[3]
            labelName = labelDirName.split('_')[0]
            label = ""
            if labelName == "deceptful":
                label = "Fake"
            else:
                label = "Real"
                
            rawData.append((text,label))
            preprocessedData.append((preProcessText(text),label))

def getLinesIndexes(reader):
    columnNames = next(reader)
    labelIndex = -1
    textIndex = -1
    for i in range(len(columnNames)):
        
        if columnNames[i] == 'LABEL' or columnNames[i] == 'Real' or columnNames[i] == 'label':
            labelIndex = i
        if columnNames[i] == 'REVIEW_TEXT' or columnNames[i] == 'Review' or columnNames[i] == 'reviewBody':
            textIndex = i
    
    if labelIndex == -1:
        raise Exception("Label index wasn't found")
    if textIndex == -1:
        raise Exception("Text index wasn't found")
    
    return (labelIndex, textIndex)

def parseReview(reviewLine, labelIndex, textIndex):
    s=""
    if reviewLine[labelIndex]=="__label1__" or reviewLine[labelIndex]==0 or reviewLine[labelIndex]=="fake":
        s = "Fake"
    else: 
        s = "Real"
    return (reviewLine[textIndex], s)

def preProcessText(reviewText):
    tokens = word_tokenize(reviewText)
    
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    return words


# =====================================================================


def convertYelpData():
    print("Conversion started")
    rx = re.compile( "^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]+.*$")
    with open("Datasets/review_data.tsv", 'r', encoding='utf-8') as myfile:  
        with open("Datasets/review_data.csv", 'w', encoding='utf-8') as csv_file:
            previousLine = None
            for line in myfile:
                if rx.match(line):
                    if not previousLine is None:
                        lineContent = list(map(quote,previousLine.split('\t')))
                        if len(lineContent)!=10:
                            print(lineContent)
                            print("------------------------")
                        else:    
                            csv_file.write(','.join(lineContent))
                        previousLine = line
                else:
                    if previousLine is None:
                        previousLine = line
                    else:    
                        previousLine = previousLine + " " + line
            
            if not previousLine is None:
                lineContent = map(quote,previousLine.split('\t'))
                csv_file.write(','.join(lineContent))
                previousLine = line
            
    print("Conversion completed")
    
def quote(s):
    if s.find(',')>=0:
        return '"'+s.replace('"','""')+'"'
    return s


# =====================================================================

def get_data_for_tokenizer_fit(dataSet):
    for texts in [x[0] for x in dataSet]:
        for text in texts:
            yield text
    for texts in [x[1] for x in dataSet]:
        for text in texts:
            yield text

def tokeniseData(dataSet):
    
    # vectorize text corpus into integers. Max size of vocabulary: 7000
    tokenizer = Tokenizer(num_words=7000)
    tokenizer.fit_on_texts( get_data_for_tokenizer_fit(dataSet) )
    
    words = [x[0] for x in dataSet]
    seq = tokenizer.texts_to_sequences(words)
    
    max_length = max([len(s) for s in seq])
    
    tokenisedData = pad_sequences(seq,padding='post', maxlen=max_length)
    
    words = [x[1] for x in dataSet]
    seq = tokenizer.texts_to_sequences(words)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    # print(X_train[2])
    # print(Tokenised_train[2])
    
    return tokenisedData, seq, vocab_size
 
def splitData(testPercentage, data, labels):
    
    # Use the last column as the target value
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=testPercentage)
    return X_train, X_test, y_train, y_test

   
def get_features_and_labels(frame):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Replace missing values with 0.0, or we can use
    # scikit-learn to calculate missing values (below)
    #frame[frame.isnull()] = 0.0

    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
    
    # Use 80% of the data for training; test against the rest
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn.pipeline.make_pipeline could also be used to chain 
    # processing and classification into a black box, but here we do
    # them separately.
    
    # If values are missing we could impute them from the training data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test


# =====================================================================


def createModel(vocab_size):
    # define model


    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, 100))
    model.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.summary()
    
    return model


def evaluate_classifier(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different classifiers to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, precision, recall)
    for each learner.
    '''

    # Import some classifiers to test
    from sklearn.svm import LinearSVC, NuSVC
    from sklearn.ensemble import AdaBoostClassifier

    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Here we create classifiers with default parameters. These need
    # to be adjusted to obtain optimal performance on your data set.
    
    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall

# =====================================================================


def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, precision, recall)
    
    All the elements in results will be plotted.
    '''

    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + URL)

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================


if __name__ == '__main__':
    # Download the amazon data set from txt file and preprocess it
    rawAmazonData = []
    preprocessedAmazonData = []
    print("Downloading amazon data from {}".format(AmazonDatasetLink))
    loadData(AmazonDatasetLink, rawAmazonData, preprocessedAmazonData, '\t')
    
    #print(preprocessedAmazonData[0])

    # Download the Yelp data set from txt file and preprocess it
    # rawYelpData = []
    # preprocessedYelpData = []
    # print("Downloading Yelp data from {}".format(YelpDatasetLink))
    
    # Only run once
    # convertYelpData
    
    # loadData(YelpDatasetLink, rawYelpData, preprocessedYelpData,',')
    
    # print(preprocessedYelpData[0])
    # print("----------------------------")

    # Download the restaurants data set from txt file and preprocess it
    # rawRestaurantData = []
    # preprocessedRestaurantData = []
    # print("Downloading restaurants data from {}".format(RestaurantDatasetLink))
    # loadData(RestaurantDatasetLink, rawRestaurantData, preprocessedRestaurantData, ',')
    
    # print(preprocessedRestaurantData[0])
    # print("----------------------------")
    
    # Download the hotels data set from folders and preprocess it
    # rawHotelData = []
    # preprocessedHotelData = []
    
    # print("Downloading restaurants data from {}".format(HotelDatasetLink))
    # prepareHotelData(rawHotelData,preprocessedHotelData)
    # print(preprocessedHotelData[0])
    # print("----------------------------")

    # Process data into feature and label arrays

    tokenisedData, tokenisedLabels, vocab_size = tokeniseData(preprocessedAmazonData)
    
    X_train, X_test, y_train, y_test = splitData(0.2, tokenisedData, tokenisedLabels)

    model = createModel(vocab_size)
    
    model.fit(tf.convert_to_tensor(X_train), y_train, epochs=10, verbose=2)

    
    # print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    # X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Evaluate multiple classifiers on the data
    # print("Evaluating classifiers")
    # results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Display the results
    # print("Plotting the results")
    # plot(results)
