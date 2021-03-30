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

total = 0
blurb_and_descr = []
with open ('init1_output2_fixed.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    for row in readCSV:
        if(row[0]=="backers_count"):
            continue
        blurb_and_descr.append(row[52])
        total = total + 1
csvfileread.close()
print("total read: ", total)


with open ("Mallet_data_prep_output.txt","w", encoding = 'Latin-1') as file1:
    for text_material in blurb_and_descr:
        stringy = "Test English " + text_material
        file1.write(stringy)
        file1.write("\n")
file1.close()