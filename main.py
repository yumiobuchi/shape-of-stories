import re
import nltk
import string
# from string import punctuation #to help remove punc
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
    orig.readline()     #to ignore first line containing headings
    words, scores, res = [],[],[]
    d = [words, scores]
    for line in orig.readlines():
        items = line.split()
        word = items[4][:-2]
        combined_emotion_score = float(items[2])-float(items[3])
        itemset = {"word":word,"score":combined_emotion_score}
        res.append(itemset)
        words.append(word)
        scores.append(combined_emotion_score)
    wr = csv.writer(final)
    wr.writerow(("word","score"))
    export_data = zip_longest(*d, fillvalue = "")
    wr.writerows(export_data)
    final.close()
    orig.close()
    return res

def convert_list_to_csv(orig, final):
    scores = [None]*len(orig)
    d = [orig, scores]  #orig is a list of words
    wr = csv.writer(final)
    wr.writerow(("word","score"))
    export_data = zip_longest(*d, fillvalue = "")
    wr.writerows(export_data)
    final.close()
    return final
def extract_features(tweet,word_feature):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 
def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

def main():
    #pre-processing: load,case, punctuation
    text = open("wuthering-heights.txt",encoding="utf-8").read() #most stuff from internet has utf-8 encoding
    translator=str.maketrans('','',string.punctuation)
    text=text.translate(translator)
    print("getting mainlines of text")
    text = get_maintext_lines_gutenberg(text)

    #tokenize and remove stop words
    testingDataSet, trainingDataSet= [], []
    for phrase in text:
        phrase_tokenized = word_tokenize(phrase)
        for word in phrase_tokenized:
            word.lower()
            if word not in stopwords.words("english") and not None:
                testingDataSet.append(word)
        print("tokenizing...")
    print("done tokenizing--------")
    #convert datasets into csv format, with 2 columns "word", "score" 
    testingDataSet = convert_list_to_csv(testingDataSet,open('tesingDataSet.csv','w',newline = ""))
    trainingDataSet = clean_up_sentiword_document(open('SentiWordNet.txt','r'),open('trainingDataSet.csv','w',newline = "" ))
    print("done converting datasets------")
    # Now we can extract the features and train the classifier 
    # UPDATE THIS. curently word_features doesnt give proper results: dict_keys(['w', 'o', 'r', 'd'])

    word_features = buildVocabulary(trainingDataSet)
    print("printing word_features-------",word_features)
    print("done printing word_features")










    
    trainingFeatures=nltk.classify.apply_features(extract_features,trainingDataSet,word_feature)

    NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
    NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0],word_feature)) for tweet in testingDataSet]

    # get the majority vote
    if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
        print("Overall Positive Sentiment")
        print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
    else: 
        print("Overall Negative Sentiment")
        print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
    #tagging emotion & split into 3 sets - train 70%, devtest %, test 20%
    # f = open("emotion.csv","r")
    # x_train, x_test,y_train, y_test = train_test_split(train_size = 0.8, shuffle = False)

        
    text.close()
    
        



    # sentiment_analyse(cleaned_text)

    # fig, axl = plot.subplots()
    # axl.bar(w.keys(),w.values())
    # fig.autofmt_xdate()
    # plot.bar(w.keys(), w.values())
    # plot.savefig('graph.png')
    # plot.show()

main()

