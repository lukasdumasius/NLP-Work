import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

idx = [] #index in file (very first line is 0 so first data is at 1)
names = []
blurbs = []
descriptions = []

total = 0
i=0 #idx starts at 0
with open ('comboC2working.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    for row in readCSV:
        category = row[2] #categories are column 2 in csv file
        if (category=="Product Design"):
            idx.append(i)
            names.append(row[22])
            blurbs.append(row[1])
            descriptions.append(row[43])

            total=total+1
        i=i+1

csvfileread.close()

blurb_and_descr = []
i=0
while i < len(blurbs):
    blurb_and_descr.append(blurbs[i] + " " + descriptions[i])
    i=i+1

i=0
for id in idx:
    print("idx: ", id)
    print("name: ", names[i])
    print("blurb and description: ", blurb_and_descr[i])
    i=i+1
print ("done")

print("total: ", total)
print("length of idx: ", len(idx))


##############################



rows=[]
row=[]
i=0
while i < len(idx):
    row=[idx[i], names[i], blurbs[i], descriptions[i], blurb_and_descr[i]]
    rows.append(list(row))
    i=i+1

with open ('init_output.csv','w', encoding="Latin-1") as csvfilewrite:
    writeCSV = csv.writer(csvfilewrite, delimiter=',', lineterminator='\n')
    for r in rows:
        writeCSV.writerow(r)
csvfilewrite.close()

