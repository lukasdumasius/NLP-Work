import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

final_score_list = []
analyzer = SentimentIntensityAnalyzer()
with open ('Kickstarter_master.csv','r') as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
     #descriptions = []
    for row in readCSV:
        description = row[10] #descriptions are row 10 in excel file
        #descriptions.append(description)

        tokenized_description = sent_tokenize(description)
        #if not tokenized_description: #use to check for empty descripts
        print(tokenized_description)


        i = 0
        total_intensity_score = 0
        for sentence in tokenized_description:
            result = analyzer.polarity_scores(sentence)
            print(result) #just debugging, can delete
            intensity_score = ((result['neg'] + result['pos'])/2) - result['neu'] #sentence intensity score
            print("Sentence ", i, "intensity score: ", intensity_score)
            total_intensity_score = total_intensity_score + intensity_score
            i = i+1
        if(i):
            total_intensity_score = total_intensity_score/i
            print("total sentences: ", i)
            print("total intensity score: ", total_intensity_score)
            final_score_list.append(total_intensity_score)
        else:
            print("Error! No description! (0 sentences counted)")
            final_score_list.append(0) #score of 0 for empty descriptions





csvfileread.close()
#print(final_score_list) debugging
print("length of final list: ", len(final_score_list))
with open ('Kickstarter_master_output.csv','w') as csvfilewrite:
    writeCSV = csv.writer(csvfilewrite, delimiter=',')
    #writeCSV.writerow(final_score_list)
    for score in final_score_list:
        writeCSV.writerow([score])
csvfilewrite.close()




################### Code for writing the last tokenized description to output file#####
#with open ('Kickstarter_master_output.csv','w') as csvfilewrite:
#    writeCSV = csv.writer(csvfilewrite, delimiter=',')
#    for sentence in tokenized_description:
#        writeCSV.writerow([sentence])
#
#csvfilewrite.close()
################### end Code for writing the last tokenized description to output file#####



