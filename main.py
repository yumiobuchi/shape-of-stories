import re
import nltk
from string import punctuation #to help remove punc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from collections import Counter
import matplotlib.pyplot as plot
import csv
from itertools import zip_longest


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

def clean_up_sentiword_document(orig, final):
    orig.readline()     #to ignore headings
    words = []
    scores = []
    d = [words, scores]
    for line in orig.readlines():
        items = line.split()
        combined_emotion_score = float(items[2])-float(items[3])
        scores.append(combined_emotion_score)
        words.append(items[4][:-2])
    wr = csv.writer(final)
    wr.writerow(("word","score"))
    export_data = zip_longest(*d, fillvalue = "")
    wr.writerows(export_data)
    orig.close()
    final.close()
    return final

def convert_list_to_csv(orig, final):
    scores = []
    d = [orig, scores]  #orig is a list of words
    




def main():
    #pre-processing: load,case, punctuation
    text = open("wuthering-heights.txt",encoding="utf-8").read() #most stuff from internet has utf-8 encoding
    translator=str.maketrans('','',string.punctuation)
    text=text.translate(translator)
    text = get_maintext_lines_gutenberg(text)

    #tokenize and remove stop words
    testingDataSet, trainingDataSet= [], []
    for phrase in text:
        phrase.lower()
        phrase_tokenized = word_tokenize(phrase)
        for word in phrase_tokenized:
            if word not in stopwords.words("english") and not None:
                testingDataSet.append(word)

    # #by now, final_words is a list of words; sentiword is a csv-- convert both of them into csv with 2 cols
    testingDataSet = convert_list_to_csv(testingDataSet,open('tesingDataSet.csv','w',newline = ""))

    #pre-processing for emotion text
    trainingDataSet = clean_up_sentiword_document(open('SentiWordNet.txt','r'),open('trainingDataSet.csv','w',newline = "" ))
    print(trainingDataSet)
    #tagging emotion & split into 3 sets - train 70%, devtest 10%, test 20%
    # f = open("emotion.csv","r")
    # x_train, x_test,y_train, y_test = train_test_split(train_size = 0.8, shuffle = False)

        
    # text.close()
    
        



    # sentiment_analyse(cleaned_text)

    # fig, axl = plot.subplots()
    # axl.bar(w.keys(),w.values())
    # fig.autofmt_xdate()
    # plot.bar(w.keys(), w.values())
    # plot.savefig('graph.png')
    # plot.show()

main()

