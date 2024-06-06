'''
This script performs the basic process for applying a machine learning
algorithm to a datasets using Python libraries.

The four steps are:
   1. Download four datasets
   2. Process the text data and tokenise it 
   3. Train and evaluate CNN-BiLSTM model
   4. Plot and compare results

============
Datasets Used
============
    1. Amazon dataset reviews include 21,000 reviews (10500 real and 10500 fake), 
       with every review includes meta-information like item Id, reviewer name, item name, 
       investigate buying (yes or no), rating value, and group label.
    2. Yelp dataset consists of conventional false electronic product evaluations from four different cities 
       in the United States (NY, Los Angeles, Miami, and San Francisco)
    3. Restaurant dataset contains 110 restaurant reviews (55 fake, 55 real)
    4. Hotel dataset contains 1600 hotel reviews (800 real and 800 fake) 
       gathered from a single of most prominent hotel booking websites Trip Advisor
    5. Combined datasets of all previous datasets
'''

# Remember to update the script for the new data when you change these links
AmazonDatasetLink = "Datasets/amazon_reviews.txt"
YelpDatasetLink = "Datasets/review_data.csv"
RestaurantDatasetLink = "Datasets/restaurant_reviews_anonymized.csv"
HotelDatasetLink = "Datasets\op_spam_v1.4"

from operator import indexOf
from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
import csv
import string
import re
import os
import pickle

import tensorflow as tf

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def loadData(path, separator):
    preprocessedData = []
    labels = []
    
    with open(path, encoding='utf-8') as f:
        
        # read file from path
        reader = csv.reader(f, delimiter = separator)
        
        # save indexes of label and text columns
        (labelIndex, textIndex) = getLinesIndexes(reader)
        
        for line in reader:
            # save review text and label
            (Text, Label) = parseReview(line, labelIndex, textIndex)
            
            preprocessedData.append(preProcessText(Text))
            labels.append(Label)
    return preprocessedData, labels
            

def prepareHotelData():
    
    # Hotel data saved as files in folders hense this method exists

    preprocessedData = []
    labels = []
    for subdir, dirs, files in os.walk(HotelDatasetLink):
        for file in files:
            f=open(os.path.join(subdir,file),'r')
            text=f.read()
            f.close()
            
            # get label from folder name
            labelDirName = subdir.split('\\')[3]
            labelName = labelDirName.split('_')[0]
            if labelName == "deceptive":
                trust = 0
            else:
                trust = 1
                
            preprocessedData.append((preProcessText(text)))
            labels.append(trust)
    return preprocessedData, labels

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
    if reviewLine[labelIndex]=="__label1__" or reviewLine[labelIndex]=="0" or reviewLine[labelIndex]=="fake":
        trust = 0 # "Fake"
    else: 
        trust = 1 # "Real"
    return (reviewLine[textIndex], trust)

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
    
    # Convert words to basic form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# =====================================================================

# Convert Yelp dataset .tsv file into .csv file. Only use once
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

# If data already exist load it from folder, otherwise tokenise, split, and save it
def prepare_data(saveFolderPath, filePath, separator):
    if os.path.isfile(saveFolderPath+"X_train.dat.npy") and \
       os.path.isfile(saveFolderPath+"X_test.dat.npy") and \
       os.path.isfile(saveFolderPath+"y_train.dat.npy") and \
       os.path.isfile(saveFolderPath+"y_test.dat.npy") and \
       os.path.isfile(saveFolderPath+"vocab_size.dat"):
        
        print(f"Loading data from cache: {saveFolderPath}")
        X_train = np.array(np.load(saveFolderPath+'X_train.dat.npy',allow_pickle=True), dtype='int')
        X_test  = np.array(np.load(saveFolderPath+'X_test.dat.npy', allow_pickle=True), dtype='int')
        y_train = np.array(np.load(saveFolderPath+'y_train.dat.npy',allow_pickle=True), dtype='int')
        y_test  = np.array(np.load(saveFolderPath+'y_test.dat.npy', allow_pickle=True), dtype='int')
        vocab_size = np.fromfile(saveFolderPath+'vocab_size.dat', dtype=int)[0]
        
    else:
        # Combine all datasets into one
        if filePath=="All":
            print("Loading data from all datasets")
            amazonPreprocessedData, amazonLabels = loadData(AmazonDatasetLink, '\t')
            yelpPreprocessedData, yelpLabels = loadData(YelpDatasetLink, separator)
            restaurantsPreprocessedData, restaurantsLabels = loadData(RestaurantDatasetLink, separator)
            hotelsPreprocessedData, hotelsLabels = prepareHotelData()
            
            preprocessedData = amazonPreprocessedData + yelpPreprocessedData + restaurantsPreprocessedData + hotelsPreprocessedData
            labels = amazonLabels + yelpLabels + restaurantsLabels + hotelsLabels
            
        else:
            print(f"Loading data from {filePath}")
            if separator == "files":
                preprocessedData, labels = prepareHotelData()
            elif separator == "tab":
                preprocessedData, labels = loadData(filePath, '\t')
            else:
                preprocessedData, labels = loadData(filePath, separator)
        
        # tokenise data to be used as input to ML model
        tokenisedData, vocab_size = tokeniseData(preprocessedData)
    
        # split data
        X_train, X_test, y_train, y_test = splitData(0.2, tokenisedData, labels)

        # save split data to folder
        np.array([vocab_size]).tofile(saveFolderPath+'vocab_size.dat')
        np.save(saveFolderPath+'X_train.dat', np.array(X_train, dtype='object'), allow_pickle=True)
        np.save(saveFolderPath+'X_test.dat' , np.array(X_test , dtype='object'), allow_pickle=True)
        np.save(saveFolderPath+'y_train.dat', np.array(y_train, dtype='object'), allow_pickle=True)
        np.save(saveFolderPath+'y_test.dat' , np.array(y_test , dtype='object'), allow_pickle=True)    

    print(f"Vocabulary size: {vocab_size}, Sentence length: {len(X_train[0])}, Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, vocab_size

# If trained model already exists load it from folder, otherwise create, train, and save it
def prepare_model(saveFolderPath, vocab_size, X_train, y_train, X_test, y_test):
    if os.path.isfile(saveFolderPath+"model1.keras"):
        model = tf.keras.models.load_model(saveFolderPath+"model1.keras")
        
        with open(saveFolderPath+'trainHistoryDict', 'rb') as file_pi:
            history = pickle.load(file_pi)
            
    else:
        model = createModel(vocab_size, len(X_train[0]))
        history = model.fit(X_train, y_train, epochs=10, verbose=2, validation_data=(X_test, y_test)).history
        
        model.save(saveFolderPath+'model1.keras')  # The file needs to end with the .keras extension
        
        with open(saveFolderPath+'trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history, file_pi)

        
    return model, history

def tokeniseData(dataSet):
    
    # vectorize text corpus into integers. Max size of vocabulary: 7000
    tokenizer =  tf.keras.preprocessing.text.Tokenizer(num_words=7000)
    tokenizer.fit_on_texts( get_data_for_tokenizer_fit(dataSet) )
    
    seq = tokenizer.texts_to_sequences(dataSet)
    
    max_length = max([len(s) for s in seq])
    
    # add padding so that all the data has the same size
    tokenisedData = tf.keras.preprocessing.sequence.pad_sequences(seq,padding='post', maxlen=max_length)
    
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    return tokenisedData, vocab_size
 
def get_data_for_tokenizer_fit(dataSet):
    for sentence in dataSet:
        for word in sentence:
            yield word

def splitData(testPercentage, data, labels):
    
    # Use the last column as the target value
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=testPercentage)
    
    X_train = np.array(X_train, dtype='int')
    X_test  = np.array(X_test , dtype='int')
    y_train = np.array(y_train, dtype='int')
    y_test  = np.array(y_test , dtype='int')

    # print("X_train={}, X_test={}, y_train={}, y_test={}".format(len(X_train), len(X_test), len(y_train), len(y_test)))
    return X_train, X_test, y_train, y_test

# =====================================================================

def createModel(vocab_size, max_len):
    # define model
    
    # CNN-BiLSTM
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(max_len,)))
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100))
    
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=False)))
    
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    
    # CNN
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Input(shape=(max_len,)))
    # model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100))
    # model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(10, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model.summary()
    
    return model

# =====================================================================

def plot(history):
    # list all data in history
    # print(history.keys())
    
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# =====================================================================

if __name__ == '__main__':
    
    # Only run once before starting working with the program
    # convertYelpData

    # Download the amazon data set from txt file and preprocess it
    X_train, X_test, y_train, y_test, vocab_size = prepare_data("Models/Amazon/",AmazonDatasetLink,"tab")
    
    model, history = prepare_model("Models/Amazon/",vocab_size,X_train,y_train,X_test,y_test)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))
    plot(history)
    
    print("----------------------------")
    
    # Download the Yelp data set from txt file and preprocess it
    X_train, X_test, y_train, y_test, vocab_size = prepare_data("Models/Yelp/",YelpDatasetLink,',')
    
    model, history = prepare_model("Models/Yelp/",vocab_size,X_train,y_train,X_test,y_test)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))
    plot(history)
    
    print("----------------------------")

    # Download the restaurants data set from txt file and preprocess it
    X_train, X_test, y_train, y_test, vocab_size = prepare_data("Models/Restaurant/",RestaurantDatasetLink,',')
    
    model, history = prepare_model("Models/Restaurant/",vocab_size,X_train,y_train,X_test,y_test)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))
    plot(history)
    
    print("----------------------------")
    
    # Download the hotels data set from folders and preprocess it
    X_train, X_test, y_train, y_test, vocab_size = prepare_data("Models/Hotel/",HotelDatasetLink, "files")
    
    model, history = prepare_model("Models/Hotel/",vocab_size,X_train,y_train,X_test,y_test)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))
    plot(history)
    
    print("----------------------------")

    # Download all datasets from folders, combine them into single dataset, and preprocess it
    X_train, X_test, y_train, y_test, vocab_size = prepare_data("Models/All/","All", ",")
    
    model, history = prepare_model("Models/All/",vocab_size,X_train,y_train,X_test,y_test)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))
    plot(history)
