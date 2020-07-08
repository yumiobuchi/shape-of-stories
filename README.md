# Motive

Inspired by Vonnegut's statement, **"There is no reason why the simple shapes of stories can’t be fed into computers, they are beautiful shapes”**, I'm interested in seeing if NLP can reveal these "shapes of stories". Specifically, I'm investigating if literary texts' sentiment time series can indicate certain literary archetypes (e.g. "Rags to riches" (rise), "Tragedy" (fall), "Cinderella" (rise-fall-rise)). 

I have since discovered a [relevant paper](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-016-0093-1) and sought to employ different techniques for classification & clustering (see Methodology)

# Methodology
Literary texts are obtained from [Project Gutenberg](https://www.gutenberg.org/)

*preprocessing*
removing Gutenberg markups, tokenizing, removing whitespace & stop words & punctuation & numbers & common names

*sentiment score*
To determine text sentiment, words are rated on a 1-9 scale from most negative to most positive. [See data source](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026752). I generate a sentiment score for every 5% of a text to get the same number of datapoints for each text. This means for each text we get an array of 20 sentiment scores. To get a time series of datapoints, we use sliding window (window size 10k words).

*classification*
Multinomial Naive Bayes

*clustering*
KMeans -- I group texts with similar "shapes" into clusters, determining a good K by elbow method. Finally we're at the fun part: generating clusters that represent archetypical emotion journeys in literature. 

# So was Kurt Vonnegut right?
Seems like it. After processing 1000 classic literary texts, we find x archetypical arcs:
INSERT GRPAH OF CLUSTERS HERE. 


# Example: a sentiment time series for Pride and Prejudice by Jane Austen
![sentiment time series](https://github.com/yumiobuchi/emotion-arcs/blob/master/sentiment_graph.png?raw=true)
Note how the "lowest" and "highest" sentiment points correlate to the most negative and positive emotional events.

# Next steps
Process more books, perhaps from a Project Gutenberg bookshelf

Smoothing the graphs

# How to run
ensure that you've pip installed relevant packages: nltk, matplotlib, numpy, pickle
run: python3 main.py



