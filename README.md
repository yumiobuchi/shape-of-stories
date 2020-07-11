# Motive

Inspired by Vonnegut's statement, **"There is no reason why the simple shapes of stories can’t be fed into computers, they are beautiful shapes”**, I'm interested in seeing if NLP can reveal these "shapes of stories". Specifically, I'm investigating if literary texts' sentiment time series can indicate certain literary archetypes (e.g. "Rags to riches" (rise), "Tragedy" (fall), "Cinderella" (rise-fall-rise)). 

I have since discovered a [relevant paper](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-016-0093-1) and sought to employ different techniques for classification & clustering (see Methodology)

# Methodology

***Training data set/How are words assigned a happiness score?***
To compare my findings with the paper, I used the same [training data](http://hedonometer.org/words/labMT-en-v1/), and recategorizing floating happiness scores to 0-9 integer scores (e.g. 8.6-> 9).

"To quantify the happiness of the atoms of language, we merged the 5,000 most frequent words from a collection of four corpora: Google Books, New York Times articles, Music Lyrics, and Twitter messages, resulting in a composite set of roughly 10,000 unique words."

***A suitable corpus:***
Literary texts are obtained from [Project Gutenberg](https://www.gutenberg.org/).
Following Regan's method, books are English, between 20k-100k owrds,and exlude those with fewer than 40 downloads. We process 1106 texts.

Scraped with this [web scraper](https://github.com/kpully/gutenberg_scraper),with some modifications

***preprocessing:***
removing Gutenberg markups, tokenizing, removing whitespace & stop words & punctuation & numbers & common names.

***sentiment score:***
To determine text sentiment, words are rated on a 1-9 scale from most negative to most positive. [See data source](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026752). I generate a sentiment score for every 5% of a text to get the same number of datapoints for each text. This means for each text we get an array of 20 sentiment scores. To get a time series of datapoints, we use sliding window (window size 10k words).

***classification:***
Multinomial Naive Bayes

***clustering:***
KMeans -- I group texts with similar "shapes" into clusters, determining a good K by elbow method, which recommends 2-3 clusters. 
![kmeans elbow method](https://github.com/yumiobuchi/emotion-arcs/blob/master/kmeans.png?raw=true)

Finally we're at the fun part: generating clusters that represent archetypical emotion journeys in literature. 

# So was Kurt Vonnegut right?
It's unclear. After processing 1106 texts, there does not appear to be major differences between the 3 clusters --i.e. we cannot group texts into literary archetypes. 

![cluster 1](https://github.com/yumiobuchi/emotion-arcs/blob/master/img/cluster_0.png?raw=true)
![cluster 2](https://github.com/yumiobuchi/emotion-arcs/blob/master/img/cluster_1.png?raw=true)
![cluster 3](https://github.com/yumiobuchi/emotion-arcs/blob/master/img/cluster_2.png?raw=true)

# How to improve
***Expand features selection:***
to include negation, humour

***Find a better training data set:***
the current training dataset contains more "modern" words and modern sentiments, as a good part of them are sourced from NYT articles and Twitter. However our testing dataset, books from Project Gutenberg are old enough that their US copyright protection has expired (age of author+70 years)

***Inherent issue with Naive Bayes classification:***
assumption of independence in features -> experiment with other classification algos


# How to run
ensure that you've pip installed relevant packages e.g. nltk, matplotlib, numpy, pickle

**run: python3 main.py**

This program with call 3 other processes to download texts, apply Naive Bayes, and cluster




