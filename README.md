# Summary

Inspired by Vonnegut's statement, **"There is no reason why the simple shapes of stories can’t be fed into computers, they are beautiful shapes”**, I'm interested in seeing if NLP can reveal these "shapes of stories". Specifically, I'm investigating if literary texts' sentiment time series can indicate certain literary archetypes (e.g. "Rags to riches" (rise), "Tragedy" (fall), "Cinderella" (rise-fall-rise)). 

I have since discovered a relevant paper and sought to employ a classificatin method different from those used in the paper: Multinomial Naive Bayes.

The [relevant paper](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-016-0093-1): Reagan, Andrew & Mitchell, Lewis & Kiley, D. & Danforth, Christopher & Dodds, Peter. (2016). The emotional arcs of stories are dominated by six basic shapes. EPJ Data Science. 5. 10.1140/epjds/s13688-016-0093-1.

# Methodology
Literary texts are obtained from [Project Gutenberg](https://www.gutenberg.org/)

In determining text sentiment, words are rated on a 1-9 scale from most negative to most positive. [See data source](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026752)

A text is preprocessed by: removing Gutenberg markups, tokenizing, and removing whitespace & stop words & punctuation & numbers.

Datapoints for sentiment time series are then obtain via a sliding window (window size 10k words).

# Example: a sentiment time series for Pride and Prejudice by Jane Austen

![sentiment time series](https://github.com/yumiobuchi/emotion-arcs/blob/master/sentiment_graph.png?raw=true)
Note how the "lowest" and "highest" sentiment points correlate to the most negative and positive emotional events.

# Next steps
Process more books, perhaps from a Project Gutenberg bookshelf

Smoothing the graphs

# How to run
ensure that you've pip installed relevant packages: nltk, matplotlib, numpy, pickle
run: python3 main.py



