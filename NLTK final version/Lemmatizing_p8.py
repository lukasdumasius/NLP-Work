from nltk.stem import WordNetLemmatizer
# use pos = "a" for words that aren't nouns
#lemmatizing is used much more than stemming probably

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))

print(lemmatizer.lemmatize("run", pos="v"))



