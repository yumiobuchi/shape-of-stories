import os

text = open("alices-adventures-in-wonderland.txt",encoding="utf-8").read() #most stuff from internet has utf-8 encoding
words = text.split()
print(len(words))