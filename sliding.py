#goal: slide every phrase
sumSoFar,dataPoint= 0,0
labels = deque()
yAxis = []
windowLen = 5000//5             #a window is 5000 words, and a phrase has roughly 5 words
for end in range(len(testingDataSet)):      #len(testingDataSet) = num of phrases
    phrase = testingDataSet[end]
    label = classifier.classify(extract_features(phrase))
    sumSoFar+=label
    labels.append(label)
    if end>=windowLen:        
        sumSoFar-=labels.popleft()          #remove datapoint
        print("labels",labels,"should be equal to",windowLen)
        dataPoint = sum(sumSoFar)/len(labels)     #update overall answer
        yAxis.append(dataPoint)
return yAxis

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