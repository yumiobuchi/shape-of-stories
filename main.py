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
    # wordCount = get_dataset_wordcount(testingDataSet)
    # print("wordCount:",wordCount)
    windowLen = 10000//5                            #10k words in a window; about 5 words in a phrase

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
            dataPoint = sumSoFar/len(labels)         #update overall answer
            sumSoFar-=labels.popleft()               #remove datapoint
            dataInit.append(dataPoint)
    numPointsInit = len(dataInit)
    step = numPointsInit//20
    for i in range(0,numPointsInit,step):
        stepAvg = sum(dataInit[i:i+step])/step
        dataFinal.append(stepAvg)
    if len(dataFinal)>20:                           #there's more often unnecessay data like index and foreword at the start 
        throw =len(dataFinal)-20
        dataFinal = datafinal[throw:]
    return dataFinal

def get_sentiment_time_series_for_text(path):
    print("path is",path)
    text = open(path,encoding="utf-8").read() 
    translator=str.maketrans('','',string.punctuation)
    text=text.translate(translator)
    print("getting mainlines of text")
    text = get_maintext_lines_gutenberg(text)                #preprocessing
    testingDataSet= preprocess(text)                         #preprocessing

    print("--testingDataSet---")
    print(testingDataSet)
    print("--end of testingDataSet---")
    trainingDataSet = preprocess_hedonometer_document(open('Hedonometer.csv'),open('trainingDataSet.csv','w',newline = "" ))
    trainingFeatures=nltk.classify.apply_features(extract_features,trainingDataSet)

    if os.path.isfile('my_classifier.pickle') and os.path.getsize('my_classifier.pickle')>0:    # checking if we've already trained the model- if not, pickle it
            print("found the model, lodading it!")
            f = open('my_classifier.pickle','rb')
            classifier = pickle.load(f)
            f.close()
    else:
        print("training the model for the first time!")
        classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
        f = open('my_classifier.pickle','wb')
        pickle.dump(classifier,f)
    return overallSentiment(classifier, testingDataSet)


def main():
    allYAxes = []
    path = os.getcwd()
    for r, d, f in os.walk(path):
        for file in fnmatch.filter(f,"*.txt"):
            print("-----now processing %s-------"%file)
            filepath = "%s/texts/%s"%(os.getcwd(),file)
            print("trying to get text from this path",filepath)
            data = get_sentiment_time_series_for_text(filepath)      #main preprocessing, training, classifying
            print(len(data),"should be 20")
            allYAxes.append(data)
    # numDataPoints = len(allYAxes[0])                              #num data points for 1 text (all texts have same # data points)
    # f = open("dataFromAllBooks.csv","w")                          #writing all NB labels of all books into a csv file                   
    # dim = ["point%s"%i for i in range(numDataPoints)]             #to create title for columns
    # with f:
    #     writer = csv.writer(f)
    #     writer.writerow(dim)
    #     writer.writerows(allYAxes)                      

    #now, apply K-means clustering to group all book time seires into clusters of similar patterns
    #plot all graphs of a similar pattern onto same chart

    ##TO DO: modify below to plot mutliple graphs onto same chart
    # print("data",data)
    # perc = np.linspace(0,100,len(data))
    # fig = plt.figure(1, (7,5))
    # textname=textname.capitalize()
    # textname = textname.replace('-',' ')
    # fig.suptitle('%s Sentiment Time Series'%textname,fontsize=13)
    # ax = fig.add_subplot(1,1,1)
    # plt.xlabel("percentage of document")
    # plt.ylabel("sentiment")

    # ymax = max(data)
    # ymin = min(data)
    # xpos_max = data.index(ymax)
    # xpos_min = data.index(ymin)
    # xmax = perc[xpos_max]
    # xmin = perc[xpos_min]

    # ax.plot(perc, data)
    # ax.annotate('Happily ever after', xy=(xmax, ymax), xytext=(xmax, ymax+0.2),size = 8)
    # ax.annotate('Infamous letter from Darcy', xy=(xmin, ymin), xytext=(xmax, ymax+0.2),size = 8)

    # fmt = '%.0f%%'
    # xticks = mtick.FormatStrFormatter(fmt)
    # ax.xaxis.set_major_formatter(xticks)

    # plt.xticks(rotation = 40)
    # plt.savefig('sentiment_graph.png')
    # plt.show()

startTime = time.time()
main()
print("----program runtime: %s minutes ----"%((time.time() - startTime)//60))
