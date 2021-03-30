


import nltk
import random
from nltk.corpus import movie_reviews

#hard to read one-liner
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#non-oneliner version below, but append throws error
#documents = []
#for category in movie_reviews.categories():
#    for fileid in movie_reviews.fileids(category):
#        documents.append(list(movie_reviews.words(fileid)), category)

random.shuffle(documents)



all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

#use top 3000 used words
word_features = list(all_words.keys())[:3000]

#input: (list of words, category)
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) #true if w is in words
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]