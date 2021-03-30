import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lxml.html.clean import clean_html
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

def cleanhtml(raw_html):
  #cleanr = re.compile('\!\[.*?\]\(.*?\)')
  #cleanr = re.compile('\!\[.*?\)')
  #cleantext = re.sub(cleanr, '', raw_html)
  no_newlines=re.sub('\n', ' ', raw_html)
  no_double_spaces = re.sub('  ', ' ', no_newlines)
  cleantext=re.sub('\!\[.*?\)', '', no_double_spaces) #cleans anything that looks like this: ![blah](blah)
  cleanertext= re.sub('#', '', cleantext)
  #cleanertext1= re.sub('\\', '', cleanertext)
  return cleanertext


categories = []
with open ('comboC2working.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    for row in readCSV:
        flag = 0
        c = row[2]
        for cat in categories:
            if (cat == c):
                flag = 1
        if(flag==0 and ( c== "Robots" or c=="Footwear" or c=="Woodworking" or c=="Product Design" or c=="Glass" or c=="Wearables" or c=="Fabrication Tools" or c=="Candles" or c=="Stationery" or c=="Ceramics")):
            categories.append(c)


csvfileread.close()

idx = [[] for k in range(len(categories))] #index in file (very first line is 0 so first data is at 1)
names = [[] for k in range(len(categories))]
blurbs = [[] for k in range(len(categories))]
descriptions = [[] for k in range(len(categories))]
blurb_and_descr = [[] for k in range(len(categories))]

total = 0
i=0 #idx

with open ('comboC2working.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    for row in readCSV:
        l=0 #counts category number
        for cat in categories:
            c = row[2] #categories are column 2 in csv file
            if (c==categories[l]):
                idx[l].append(i)
                names[l].append(row[22])
                blurbs[l].append(row[1])
                descriptions[l].append(row[43])
                total=total+1
            l=l+1
        i = i+1

csvfileread.close()


l=0
for cat in categories:
    i=0
    while i < len(blurbs[l]):
        blurb_and_descr[l].append(blurbs[l][i] + " " + descriptions[l][i])
        ##html cleaner##
        #document = lxml.html.document_fromstring(blurb_and_descr[l][i])
        #raw_text = document.text_content()
        #blurb_and_descr[l][i]=raw_text
        ##html cleaner end##
        #regex (new) html cleaner#
        blurb_and_descr[l][i]=cleanhtml(blurb_and_descr[l][i])
        #end regex (new) html cleaner#
        i=i+1
    l=l+1


l=0
for cat in categories:
    i=0
    for id in idx[l]:
        print("category: ", cat)
        print("idx: ", id)
        print("name: ", names[l][i])
        print("blurb and description: ", blurb_and_descr[l][i])
        i=i+1

    l=l+1
print ("done")

print("total: ", total)
print("individual category totals: ")
l=0
for cat in categories:
   print("cat = ", cat)
   print(len(idx[l]))
   l=l+1

print(".... lengths of idx, names, blurbs, description, blurb + descr: ")
print(len(idx))
print(len(names))
print(len(blurbs))
print(len(descriptions))
print(len(blurb_and_descr))


##### vader + smog stuff ####
final_score_list = []
word_count_list = []
analyzer = SentimentIntensityAnalyzer()

smog_word_count_list = []
smog_readability = []
for category in blurb_and_descr:
    i = 0
    for placeholder in category:
        tokenized_description = sent_tokenize(category[i])

        k = 0
        total_intensity_score = 0
        word_count = 0
        for sentence in tokenized_description:
            result = analyzer.polarity_scores(sentence)
            print(result)  # just debugging, can delete
            intensity_score = ((result['neg'] + result['pos']) / 2) # sentence intensity score
            print("Sentence ", k, "intensity score: ", intensity_score)
            total_intensity_score = total_intensity_score + intensity_score
            k = k + 1

            ##word counter###
            num_of_words = len(word_tokenize(sentence))
            word_count = word_count + num_of_words
        word_count_list.append(word_count)
        if (k):
            total_intensity_score = total_intensity_score / k
            print("total sentences: ", k)
            print("total intensity score: ", total_intensity_score)
            final_score_list.append(total_intensity_score)
        else:
            print("Error! No description! (0 sentences counted)")
            final_score_list.append(0)  # score of 0 for empty descriptions

        smog_word_count_list.append(textstat.lexicon_count(category[i], removepunct=True))
        smog_readability.append(textstat.smog_index(category[i]))
        i=i+1

##### end vader stuff ####


#prints blurb + descr of single (adapt for first/all?) category type
rows=[]
row=[]
l=0
for placeholder1 in idx:
    i = 0
    for placeholder2 in placeholder1:
        row=[categories[l], idx[l][i], names[l][i], blurbs[l][i], descriptions[l][i], blurb_and_descr[l][i], final_score_list[l+i], word_count_list[l+i], smog_word_count_list[l+i], smog_readability[l+i]]
        rows.append(list(row))
        i=i+1
    l=l+1




with open ('init1_output0.csv','w', encoding="Latin-1") as csvfilewrite:
    writeCSV = csv.writer(csvfilewrite, delimiter=',', lineterminator='\n')
    for r in rows:
        writeCSV.writerow(r)
csvfilewrite.close()

