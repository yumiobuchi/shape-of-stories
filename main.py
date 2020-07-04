import re
import nltk
import string
import pickle
# from string import punctuation #to help remove punc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from collections import Counter
import matplotlib.pyplot as plot
import csv
from itertools import zip_longest
import os
import time
from matplotlib import pyplot as plt

def get_maintext_lines_gutenberg(text): #from https://github.com/andyreagan/core-stories/blob/master/src/bookclass.py
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
        print(row[1],":",row[3])
        if count==2:
            break
        count+=1

def recategorize(origscore):     #recategorizes Hedonometer.csv's granular data to 9 categories
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
        
def preprocess_hedonometer_document(orig, final):
    words,sents,res = [],[],[]
    d = [words,sents]
    reader = csv.reader(orig,delimiter=",")
    firstrow = next(reader)         #first row is column headings
    try:
        for row in reader:
            print(row)
            row = next(reader)
            word = row[1]
            sent = row[3]
            stddev = row[4]
            if float(stddev)>2:                #ignore words with overly high sentiment std dev 
                continue
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

def preprocess(text):
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
        phrase_final = []           #reset
        print("tokenizing...")
    return res
def extract_features(phrase):
    return {word: True for word in phrase} 
def overallSentiment(classifier,text,start,end):
    print("sentiment from start:",start,"to end",end)
    labels = [classifier.classify(extract_features(phrase)) for phrase in text[start:end]]
    res = sum(labels)/len(labels)   #get the majority vote
    print(res)
    if res>5:
        print("Overall Positive Sentiment")
    elif res ==5:
        print("Overall Neutral Sentiment")
    else:
        print("Overall Negative Sentiment")
        # print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count("neg")/len(NBResultLabels)) + "%")
    return res

def main():
    #pre-processing: load,case, punctuation
    text = open("sentence.txt",encoding="utf-8").read() #most stuff from internet has utf-8 encoding
    translator=str.maketrans('','',string.punctuation)
    text=text.translate(translator)
    print("getting mainlines of text")
    text = get_maintext_lines_gutenberg(text)
    print(text)

    #tokenize and remove stop words
    testingDataSet= preprocess(text)
    trainingDataSet = preprocess_hedonometer_document(open('Hedonometer.csv'),open('trainingDataSet.csv','w',newline = "" ))
    trainingFeatures=nltk.classify.apply_features(extract_features,trainingDataSet)

    # checking if we've already trained the model- if not, pickle it
    if os.path.isfile('my_classifier.pickle'):
        print("found the model, lodading it!")
        f = open('my_classifier.pickle','rb')
        classifier = pickle.load(f)
        f.close()
    else:
        print("training the model for the first time!")
        classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
        f = open('my_classifier.pickle','wb')
        pickle.dump(classifier,f)
    window = 20
    numphrases = 20/5
    
    # window = 5000               #number of words to plot 1 sentiment datapoint in graph
    # numphrases = 5000//22       #around 22 words per sentence
    start,end, datapoint = 0,0,0
    yAxis,xAxis = [],[]
    for phrase in testingDataSet:   #sliding window 
        end +=1
        if end %numphrases==0:      #end of window
            datapoint = overallSentiment(classifier,testingDataSet,start,end)
            yAxis.append(datapoint)
            xAxis.append(end*5)
            # xAxis.append(end*22)
            start =end
    plt.plot(xAxis,yAxis)
    print("--------",len(xAxis),len(yAxis),"-------")
    plt.ylim(ymin=0,ymax=9)
    plt.xlim(xmin=0)
    plt.xlabel("number of words")
    plt.ylabel("sentiment")
    plt.savefig('sentiment_graph.png')
    plt.show()

startTime = time.time()
main()
print("----program initial run: %s hours ----"%((time.time() - startime)//3600))