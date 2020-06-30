import string   #to help remove punc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from collections import Counter
import matplotlib.pyplot as plot

def preprocess_sentiment_dict():
    f = open("SentiWordNet.txt",encoding="utf-8")
    # line = f.readline()
    # while line:
    #     line = f.readline()
    #     print(line)
    #     break
    # cleaned = 
    f.close()


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

def main():
    #STEP 1 pre-processing: load,case, punctuation
    text = open("wuthering-heights.txt",encoding="utf-8").read() #most stuff from internet has utf-8 encoding
    translator=str.maketrans('','',string.punctuation)
    text=text.translate(translator)
    text = get_maintext_lines_gutenberg(text)
    cleaned_text = [x.lower() for x in text]

    # #STEP 2 remove stop_words
    final_words = []
    for i in cleaned_text:
        if i not in stopwords.words("english") and not "":
            final_words.append(i)
    print(final_words)

    # #STEP 3 Bag of words algo
    # #check if words in final_words are in emotions
    # #if it's present, add to emotion_list
    # #count each emotion in emotion_list
    # emotion_list = []
    # with open('emotion.txt','r') as f:
    #     for line in f:
    #         clear_line = line.replace("\n","").replace(",","").replace("'","").strip()
    #         word,emotion = clear_line.split(": ")

    #         if word in final_words:
    #             emotion_list.append(word)
    # w = Counter(emotion_list)
    # print(w)

    # def sentiment_analyse(cleaned_text):
    #     score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    #     neg = score['neg']
    #     pos = score['pos']
    #     if neg> pos:
    #         print("negative sentiment")
    #     elif pos>neg:
    #         print("positive sentiment")
    #     else:
    #         print("neutral")
    #     print(score)

    # sentiment_analyse(cleaned_text)

    # fig, axl = plot.subplots()
    # axl.bar(w.keys(),w.values())
    # fig.autofmt_xdate()
    # plot.bar(w.keys(), w.values())
    # plot.savefig('graph.png')
    # plot.show()

main()

