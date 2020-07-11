import re
import nltk
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import csv
from itertools import zip_longest
import os
import time
from collections import deque
import fnmatch

def recategorize(origscore):                    #recategorizes Hedonometer.csv's granular data to 9 categories
    res = 0
    if origscore<1.5:
        res = 1
    elif origscore<2.5:
        res = 2
    elif origscore<3.5:
        res = 3
    elif origscore<4.5:
        res = 4
    elif origscore<5.5:
        res = 5
    elif origscore<6.5:
        res = 6
    elif origscore<7.5:
        res = 7
    elif origscore<8.5:
        res = 8
    else:
        res = 9
    return res
def get_common_names():
    reader = csv.reader(open('names.csv'),delimiter=",")
    nameset = set()
    try:
        for row in reader:
            nameset.add(row[0])
            nameset.add(row[1])
            row = next(reader)
    except:                                   #ignore StopIteration exception that arises when using next(reader)
        pass
    return nameset
        
def preprocess_hedonometer_document(orig, final):
    words,sents,res = [],[],[]
    d = [words,sents]
    reader = csv.reader(orig,delimiter=",")
    firstrow = next(reader)                    #first row is column headings
    commonNames = get_common_names()
    try:
        for row in reader:
            row = next(reader)
            word = row[1]
            if word in commonNames:            
                continue
            sent = row[3]
            stddev = row[4]
            sent = recategorize(float(sent))   #make sentiment categories less granular  
            itemset = {"word":word,"sentiment":sent}
            res.append(([word],sent))
            words.append(word)
            sents.append(sent)
    except:                                     #ignore StopIteration exception that arises when using next(reader)
        pass
    wr = csv.writer(final)
    wr.writerow(("word","sentiment"))
    export_data = zip_longest(*d, fillvalue = "")
    wr.writerows(export_data)
    final.close()
    orig.close()
    return res

def get_maintext_lines_gutenberg(text):         #this function is from https://github.com/andyreagan/core-stories/blob/master/src/bookclass.py
    lines = text.split("\n")
    start_book_i = 0
    end_book_i = len(lines)-1
    start1="START OF THIS PROJECT GUTENBERG EBOOK"
    start2="START OF THE PROJECT GUTENBERG EBOOK"
    end1="END OF THIS PROJECT GUTENBERG EBOOK"
    end2="END OF THE PROJECT GUTENBERG EBOOK"
    end3="END OF PROJECT GUTENBERG"
    for j,line in enumerate(lines):
        if (start1 in line) or (start2 in line):
            # and "***" in line and start_book[i] == 0 and j<.25*len(lines):
            start_book_i = j
        end_in_line = end1 in line or end2 in line or end3 in line.upper()
        if end_in_line and (end_book_i == (len(lines)-1)):
            #  and "***" in line and j>.75*len(lines)
            end_book_i = j
    # pass 2, this will bring us to 99%
    if (start_book_i == 0) and (end_book_i == len(lines)-1):
        for j,line in enumerate(lines):
            if ("end" in line.lower() or "****" in line) and  "small print" in line.lower() and j<.5*len(lines):
                start_book_i = j
            if "end" in line.lower() and "project gutenberg" in line.lower() and j>.75*len(lines):
                end_book_i = j
        # pass three, caught them all (check)
        if end_book_i == len(lines)-1:
            for j,line in enumerate(lines):
                if "THE END" in line and j>.9*len(lines):
                    end_book_i = j
    return lines[(start_book_i+1):(end_book_i)]

    reader = csv.reader(f,delimiter=",")
    count = 0
    for row in reader:
        if count==2:
            break
        count+=1

def preprocess(text):                              #tokenizing, lower case, removing stop words & punctuations & space & numbers
    print("preprocessing testdata")
    res,phrase_final = [],[]
    for phrase in text:
        if phrase.isspace() or phrase =="":
            continue
        phrase_tokenized = word_tokenize(phrase)
        for word in phrase_tokenized:
            word = word.lower()
            if word != "" and not word.isspace() and word.isalpha() and word not in stopwords.words("english"):
                phrase_final.append(word)
            else:
                continue
        if len(phrase_final)==0:
            continue
        res.append(phrase_final)
        phrase_final = []                           #reset
        print("tokenizing...")
    return res
def extract_features(phrase):
    return {word: True for word in phrase} 
def get_dataset_wordcount(dataset):
    count =0
    for i in dataset:
        count+=len(i)
    return count

def overallSentiment(classifier,testingDataSet):    #use sliding window (10000 words) to get graph datapoints 
    sumSoFar,wordsSoFar= 0,0
    labels = deque()
    dataInit,dataFinal = [],[]

    windowLen = 5000//5                            #5k words in a window; about 5 words in a phrase
    try:
        for end in range(len(testingDataSet)):          #len(testingDataSet) = num of phrases
            phrase = testingDataSet[end]
            wordsSoFar += len(phrase)
            label = classifier.classify(extract_features(phrase))
            sumSoFar+=label
            labels.append(label)
            if label <=1:                               #remove extreme values
                sumSoFar-=label
                labels.pop()
                continue
            if end>=windowLen-1:
                dataPoint = sumSoFar/len(labels)         #update overall answer and slide the window
                sumSoFar-=labels.popleft()               
                dataInit.append(dataPoint)
        numPointsInit = len(dataInit)
        print("numPoinstInit",numPointsInit)
        if numPointsInit<20:
            raise Exception ("Book is too short to generate enough data points")
        step = numPointsInit//20
        for i in range(0,numPointsInit,step):           #we take 20 steps(get 20 datapoints per book), by taking the avg of the datapoints within a step
            stepAvg = sum(dataInit[i:i+step])/step
            dataFinal.append(stepAvg)
        if len(dataFinal)>20:                           #usually off by 1 or 2; discard front. There's more often unnecessay data like index and foreword/acknowledgements at the the 2 ends 
            throw =len(dataFinal)-20
            dataFinal = dataFinal[throw-1:-1]
        print("the final array of sentiments for this book",dataFinal)
        return dataFinal
    except Exception as e:
        print("Exception occured in overallSentiment")
        print(e)
        pass

def get_sentiment_time_series_for_text(path):
    try:
        print("path is",path)
        text = open(path,encoding="utf-8").read() 
        translator=str.maketrans('','',string.punctuation)
        text=text.translate(translator)
        print("getting mainlines of text")
        text = get_maintext_lines_gutenberg(text)                #preprocessing
        testingDataSet= preprocess(text)                         #preprocessing

        if os.path.isfile('my_classifier.pickle') and os.path.getsize('my_classifier.pickle')>0:    # checking if we've already trained the model- if not, pickle it
                print("found the already-trained model, lodading it!")
                f = open('my_classifier.pickle','rb')
                classifier = pickle.load(f)
                classifier.show_most_informative_features(20)
                f.close()
        else: 
            trainingDataSet = preprocess_hedonometer_document(open('Hedonometer.csv'),open('trainingDataSet.csv','w',newline = "" ))
            trainingFeatures=nltk.classify.apply_features(extract_features,trainingDataSet)
            print("training the model for the first time!")
            classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
            f = open('my_classifier.pickle','wb')
            pickle.dump(classifier,f)
        return overallSentiment(classifier, testingDataSet)
    except Exception as e:
        print("exception occured in get_sentiment_time_series_for_text")
        print(e)
        pass

def main():
    #apply Naive Bayes classification to all texts
    allData = []                                                    #nested lists. Each list is a book's sentiment scores
    path = os.getcwd()
    for r, d, f in os.walk(path):
        for file in fnmatch.filter(f,"*.txt"):
            if file!="bookids.txt":
                try:
                    print("-----now processing %s-------"%file)
                    filepath = "%s/temptexts/%s"%(os.getcwd(),file)
                    data = get_sentiment_time_series_for_text(filepath)      #main preprocessing, training, classifying
                    if data is None:
                        print("data is none, ----abandoning %s----"%file)
                        continue
                    else:
                        filename = file.replace("-"," ").capitalize()
                        filename = filename[:-4]
                        data.append(filename)
                        allData.append(data)
                except:
                    print("-----abandoning %s-----"%file)

    #get data ready for generating KMeans clusters for all texts
    dataForClustering = open("dataForClustering.csv","w")
    wr = csv.writer(dataForClustering)
    firstrow = []
    numPoints = len(allData[0])
    for i in range(numPoints):
        if i==numPoints-1:
            firstrow.append("bookname")
        else:
            firstrow.append( "point %d"%(i+1))
    wr.writerow(firstrow)
    wr.writerows(allData)
    dataForClustering.close()

startTime = time.time()
main()
print("----program runtime: %s minutes ----"%((time.time() - startTime)//60))
