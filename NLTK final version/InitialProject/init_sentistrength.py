import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#####new code######
import subprocess
import shlex
import os.path
import sys

SentiStrengthLocation = "C:\\Users\\Admin\\Desktop\\NLTK\\InitialProject\\SentiStrength\\SentiStrengthCom.jar" #The location of SentiStrength
SentiStrengthLanguageFolder = "C:\\Users\\Admin\\Desktop\\NLTK\\InitialProject\\SentiStrength\\SentiStrength_DataEnglishFeb2017\\" #The location of the unzipped SentiStrength data files

## tests if sentistrength files found properly
if not os.path.isfile(SentiStrengthLocation):
    print("SentiStrength not found at: ", SentiStrengthLocation)
if not os.path.isdir(SentiStrengthLanguageFolder):
    print("SentiStrength data folder not found at: ", SentiStrengthLanguageFolder)

def RateSentiment(sentiString):
    #open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(shlex.split("java -jar '" + SentiStrengthLocation + "' stdin sentidata '" + SentiStrengthLanguageFolder + "'"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    b = bytes(sentiString.replace(" ","+"), 'utf-8') #Can't send string in Python 3, must send bytes
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")  #convert from byte
    stdout_text = stdout_text.rstrip().replace("\t"," ") #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1 -5
    return stdout_text + " " + sentiString

#### end new code ####


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
        ###modified code ###

        for sentence in tokenized_description:
            result = RateSentiment(sentence)
            print(result) #just debugging, can delete
            intensity_list_temp = [] # first item is value of pos, second is neg
            k = 0
            for j in result:
                if(k== 0 or k==3):
                    intensity_list_temp.append(int(j))
                #if(k==3):
                #    break
                k = k + 1

            print("list for debug: ") #delete
            print(intensity_list_temp) #delete
            #k=0
            #for j in intensity_list_temp: #take abs values, not needed though
            #    if(j<0):
            #        intensity_list_temp[k] = intensity_list_temp[k]*(-1)
            #    k=k+1

            intensity_score = ((intensity_list_temp[0] + intensity_list_temp[1])/2) #sentence intensity score
            print("Sentence ", i, "intensity score: ", intensity_score)
            total_intensity_score = total_intensity_score + intensity_score
            i = i+1
        ###end modified code ###

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
with open ('kickstarter_master_output_sentistrength.csv','w') as csvfilewrite:
    writeCSV = csv.writer(csvfilewrite, delimiter=',')
    #writeCSV.writerow(final_score_list)
    for score in final_score_list:
        writeCSV.writerow([score])
csvfilewrite.close()




################### Code for writing the last tokenized description to output file#####
#with open ('kickstarter_master_output_sentistrength.csv','w') as csvfilewrite:
#    writeCSV = csv.writer(csvfilewrite, delimiter=',')
#    for sentence in tokenized_description:
#        writeCSV.writerow([sentence])
#
#csvfilewrite.close()
################### end Code for writing the last tokenized description to output file#####



