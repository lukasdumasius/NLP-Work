import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lxml.html.clean import clean_html
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from textblob import TextBlob
from scipy.stats import skew, pearsonr
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import string

total = 0
categories = {}
with open ('comboC2working.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    for row in readCSV:
        c=row[2]
        flag = 0
        for category in categories.keys():
            if (c == category):
                categories[c] = categories[c]+ 1
                flag = 1
                break
        if(flag == 0):
            categories[c] = 1



file1 = open("category_totals.txt", "w")
file1.write("Category totals\n")
for category in categories.keys():
    file1.write( "Category: " + str(category) + " quantity: " + str(categories[category]) + "\n")
    total = total + categories[category]
file1.write("Total entries: " + str(total))
file1.close()