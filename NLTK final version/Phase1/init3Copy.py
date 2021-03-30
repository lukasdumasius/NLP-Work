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

def cleanhtml(raw_html):
  #cleanr = re.compile('\!\[.*?\]\(.*?\)')
  #cleanr = re.compile('\!\[.*?\)')
  #cleantext = re.sub(cleanr, '', raw_html)
  no_newlines=re.sub('\n', ' ', raw_html)
  #no_double_spaces = re.sub('  ', ' ', no_newlines)
  cleantext=re.sub('\!\[.*?\]\(.*?\)', '', no_newlines) #cleans anything that looks like this: ![blah](blah)
  cleanertext = re.sub("([a-z]([a-z]|\d|\+|-|\.)*):(\/\/(((([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:)*@)?((\[(|(v[\da-f]{1,}\.(([a-z]|\d|-|\.|_|~)|[!\$&'\(\)\*\+,;=]|:)+))\])|((\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]))|(([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=])*)(:\d*)?)(\/(([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*|(\/((([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)+(\/(([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*)?)|((([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)+(\/(([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*)|((([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)){0})(\?((([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|[\xE000-\xF8FF]|\/|\?)*)?(\#((([a-z]|\d|-|\.|_|~|[\x00A0-\xD7FF\xF900-\xFDCF\xFDF0-\xFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|\/|\?)*)?", '', cleantext)
  cleanesttext= re.sub('#', '', cleanertext)

  allow = string.ascii_letters + string.digits + string.punctuation + string.whitespace
  cleanestesttext = re.sub('[^%s]' % allow, '', cleanesttext)

  no_extra_spaces = re.sub('\s\s+', ' ', cleanestesttext)
  no_html_warning = re.sub('You\'ll need an HTML5 capable browser to see this content\.', '', no_extra_spaces)
  no_youtube_warning = re.sub(r"Play Replay with sound Play with sound 00:00 00:00", '', no_html_warning)
  no_extra_spaces_2 = re.sub('\s\s+', ' ', no_youtube_warning)

  return no_extra_spaces_2

textblob_polarity_list=[]
textblob_subjectivity_list=[]
def textblob_implemenation():
    i=0
    for placeholder5 in blurb_and_descr:
        for placeholder6 in placeholder5:
            temp_textblob = 0
            temp_textblob_s = 0

            sentences = sent_tokenize(placeholder6)
            k=0
            for sentence in sentences:
                blob = TextBlob(sentence)
                sentiment = blob.sentiment

                sentence_length = len(word_tokenize(sentence))
                temp_textblob = temp_textblob + (sentiment[0])
                temp_textblob_s = temp_textblob_s + (sentiment[1])
                k=k+1

            if (k):
                textblob_polarity_list.append(temp_textblob/k)
                textblob_subjectivity_list.append(temp_textblob_s/k)
            else:
                print("textblob - Error! No description! (0 sentences counted)")
                textblob_polarity_list.append(9)
                textblob_subjectivity_list.append(9)
            i=i+1

no_outliers_list_vader = []
no_outliers_list_textblob_polarity = []
no_outliers_list_textblob_subjectivity = []
logged_list_vader = []
logged_list_textblob_polarity = []
logged_list_textblob_subjectivity = []
temp_id_4=[]
no_errors_no_outliers_list_vader = []
no_errors_no_outliers_list_textblob_polarity = []
no_errors_no_outliers_list_textblob_subjectivity = []
no_errors_logged_list_vader = []
no_errors_logged_list_textblob_polarity = []
no_errors_logged_list_textblob_subjectivity = []



everything_included_outlier_list =[]

#temp_ids refer to ids a listed in the modified list of entries of interest
#idx array refers to the "REAL" index in the original file
def normalization_and_graphing():
    ###probably useful data strucutures for this function:
    # vader_polarity_list
    # textblob_polarity_list
    # final_idx_list #Not used within function, but could be useful
    ##################

    global everything_included_list_vader
    everything_included_list_vader = np.log1p(vader_polarity_list)
    global everything_included_list_textblob_polarity
    everything_included_list_textblob_polarity = np.log1p(textblob_polarity_list)
    global everything_included_list_textblob_subjectivity
    everything_included_list_textblob_subjectivity = np.log1p(textblob_subjectivity_list)


    skew_textblob = skew(textblob_polarity_list)
    skew_vader = skew(vader_polarity_list)
    skew_textblob_subjectivity = skew(textblob_subjectivity_list)
    print("length of final score list: ", len(vader_polarity_list))
    print("length of textblob final score list: ", len(textblob_polarity_list))
    print("length of textblob subjectivity final score list: ", len(textblob_subjectivity_list))
   
    print("skew vader: ", skew_vader)
    print("skew textblo_polarityb: ", skew_textblob)
    print("skew textblob_subjectivity: ", skew_textblob_subjectivity)


    plt_1 = plt.figure(1)
    plt.hist(vader_polarity_list, bins=80, range=[-1, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('vader_polarity - original')
    plt.show()
    input("Press Enter to continue...")

    plt_2 = plt.figure(2)
    plt.hist(textblob_polarity_list, bins=80, range=[-1, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('textblob_polarity - original')
    plt.show()
    input("Press Enter to continue...")

    plt_3 = plt.figure(3)
    plt.hist(textblob_subjectivity_list, bins=80, range=[0, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('textblob_subjectivity - original')
    plt.show()
    input("Press Enter to continue...")

    print('removing outliers vader')

    elements = np.array(vader_polarity_list)

    mean = np.mean(elements)
    sd = np.std(elements)

    temp_id = []
    current_id=0
    for x in vader_polarity_list:
        if(x > mean - 3 * sd and x < mean + 3 * sd):
             ignore = 1 #does nothing
        else:
            temp_id.append(current_id)
        current_id = current_id + 1

    temp_id_1 = temp_id.copy()



    print('removing outliers textblob_polarity')

    elements = np.array(textblob_polarity_list)

    mean = np.mean(elements)
    sd = np.std(elements)

    temp_id = []
    current_id = 0
    for x in textblob_polarity_list:
        if (x > mean - 3 * sd and x < mean + 3 * sd):
            ignore = 1 #does nothing
        else:
            temp_id.append(current_id)
        current_id = current_id + 1

    temp_id_2 = temp_id.copy()

    print('removing outliers textblob_subjectivity')

    elements = np.array(textblob_subjectivity_list)

    mean = np.mean(elements)
    sd = np.std(elements)

    temp_id = []
    current_id = 0
    for x in textblob_subjectivity_list:
        if (x > mean - 3 * sd and x < mean + 3 * sd):
            ignore = 1  # does nothing
        else:
            temp_id.append(current_id)
        current_id = current_id + 1

    temp_id_3 = temp_id.copy()

    temp_id_4 = temp_id_1 + temp_id_2 + temp_id_3 #temp_id_4 now contains all outlier ids

    
    current_id=0
    flag=0
    for placeholder7 in vader_polarity_list:
        for placeholder8 in temp_id_4:
            if(current_id == placeholder8):
                flag = 1
                break
        if (flag == 0):
            no_outliers_list_vader.append(vader_polarity_list[current_id])
            no_outliers_list_textblob_polarity.append(textblob_polarity_list[current_id])
            no_outliers_list_textblob_subjectivity.append(textblob_subjectivity_list[current_id])

            no_errors_no_outliers_list_vader.append(vader_polarity_list[current_id])
            no_errors_no_outliers_list_textblob_polarity.append(textblob_polarity_list[current_id])
            no_errors_no_outliers_list_textblob_subjectivity.append(textblob_subjectivity_list[current_id])

            everything_included_outlier_list.append("not outlier")

        if(flag == 1):
            no_outliers_list_vader.append("Outlier ERROR- YOU SHOULD NEVER SEE THIS")
            no_outliers_list_textblob_polarity.append("Outlier ERROR- YOU SHOULD NEVER SEE THIS")
            no_outliers_list_textblob_subjectivity.append("Outlier ERROR- YOU SHOULD NEVER SEE THIS")

            everything_included_outlier_list.append("outlier")


        flag = 0
        current_id = current_id + 1

    print("length of vader_polarity - outliers removed: ", len(no_errors_no_outliers_list_vader))
    print("length of textblob_polarity - outliers removed: ", len(no_errors_no_outliers_list_textblob_polarity))
    print("length of textblob_subjectivity - outliers removed: ", len(no_errors_no_outliers_list_textblob_subjectivity))


    print("outliers removed vader_polarity skew: ", skew(no_errors_no_outliers_list_vader))
    print("outliers removed textblob_polarity skew: ", skew(no_errors_no_outliers_list_textblob_polarity))
    print("outliers removed textblob_subjectivity skew: ", skew(no_errors_no_outliers_list_textblob_subjectivity))

    plt_4 = plt.figure(4)
    plt.hist(no_errors_no_outliers_list_vader, bins=80, range=[-1, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('vader_polarity - outliers removed')
    plt.show()
    input("Press Enter to continue...")

    plt_5 = plt.figure(5)
    plt.hist(no_errors_no_outliers_list_textblob_polarity, bins=80, range=[-1, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('textblob_polarity - outliers removed')
    plt.show()
    input("Press Enter to continue...")

    plt_6 = plt.figure(6)
    plt.hist(no_errors_no_outliers_list_textblob_subjectivity, bins=80, range=[0, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('textblob_subjectivity - outliers removed')
    plt.show()
    input("Press Enter to continue...")

    print("MAKE SURE THESE ARE SAME VALUE: ")
    print(len(no_outliers_list_vader))
    print(len(no_outliers_list_textblob_polarity))
    print(len(no_outliers_list_textblob_subjectivity))

    print("and these: ")



    print("Finding log of both score lists...")

    no_errors_logged_list_vader = np.log1p(no_errors_no_outliers_list_vader)
    no_errors_logged_list_textblob_polarity = np.log1p(no_errors_no_outliers_list_textblob_polarity)
    no_errors_logged_list_textblob_subjectivity = np.log1p(no_errors_no_outliers_list_textblob_subjectivity)


    i=0
    for placeholder10 in no_outliers_list_vader:
        if(isinstance(placeholder10, str)):
            logged_list_vader.append("outlier")
            logged_list_textblob_polarity.append("outlier")
            logged_list_textblob_subjectivity.append("outlier")

        else:
            logged_list_vader.append(math.log1p(no_outliers_list_vader[i]))
            logged_list_textblob_polarity.append(math.log1p(no_outliers_list_textblob_polarity[i]))
            logged_list_textblob_subjectivity.append(math.log1p(no_outliers_list_textblob_subjectivity[i]))

        i=i+1

    plt_7 = plt.figure(7)
    plt.hist(no_errors_logged_list_vader, bins=80, range=[-1, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('vader_polarity - normalized')
    plt.show()
    input("Press Enter to continue...")

    plt_8 = plt.figure(8)
    plt.hist(no_errors_logged_list_textblob_polarity, bins=80, range=[-1, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('textblob_polarity - normalized')
    plt.show()
    input("Press Enter to continue...")

    plt_9 = plt.figure(9)
    plt.hist(no_errors_logged_list_textblob_subjectivity, bins=80, range=[0, 1], align='mid')
    plt.ylabel("Frequency")
    plt.title('textblob_subjectivity - normalized')
    plt.show()
    input("Press Enter to continue...")

    print("Finding The Pearson Correlation Coefficient between vader_polarity scores and textblob_polarity scores...")
    print("Pearson Correlation Coefficient: ", pearsonr(no_errors_logged_list_textblob_polarity, no_errors_logged_list_vader))


valence_list =[]
def simple_valence_calculator():
    i=0
    for id in final_idx_list:
        flag1 = 0
        if(vader_polarity_list[i]>=0 and textblob_polarity_list[i]>=0):
            valence_result = 0 #positive
            flag1 = 1

        if (vader_polarity_list[i] <= 0 and textblob_polarity_list[i] <= 0):
            valence_result = 1  # positive
            flag1 = 1
        if( (vader_polarity_list[i]>=0 and textblob_polarity_list[i] <= 0) or (vader_polarity_list[i]<=0 and textblob_polarity_list[i]>=0) ):
            valence_result = 0.5 #conflicting scores
            flag1 = 1

        if(flag1==0):
            valence_result = -1 #error code, shouldnt happen

        valence_list.append(valence_result)
        i = i + 1

duration_list =[]
def duration_calculation():
    i=0
    for id  in final_idx_list:
        try:
            duration_string_temp_list_start = date_start[i].split('/')
            duration_string_temp_list_end = date_end[i].split('/')

            date_a = int(duration_string_temp_list_start[2])
            date_b = int(duration_string_temp_list_start[0])
            date_c = int(duration_string_temp_list_start[1])
            date_d = int(duration_string_temp_list_end[2])
            date_e = int(duration_string_temp_list_end[0])
            date_f = int(duration_string_temp_list_end[1])

            d1 = datetime.date(date_a, date_b, date_c)
            d2 = datetime.date(date_d, date_e, date_f)
            duration_result_temp = (d2-d1).days

            duration_list.append(duration_result_temp)
        except:
            duration_list.append(-1) #error code, invalid date
            print("Invalid date start/end, appending '-1' for duration")

        i=i+1


sadness_score_list = []
joy_score_list = []
anger_score_list = []
fear_score_list = []
def nrc_intensity_processing():
    for category in blurb_and_descr:
        for entry in category:
            words = []
            sentences = sent_tokenize(entry)
            for sentence in sentences:
                words_temp = word_tokenize(sentence)
                for word in words_temp:
                    if(word.isalpha()):
                        words.append(word)

            for word in words:
                word_lowercase = word.lower()

                sadness_score = 0
                joy_score = 0
                anger_score = 0
                fear_score = 0
                
                sadness_wordcount = 0
                joy_wordcount = 0
                anger_wordcount = 0
                fear_wordcount = 0

                nrc_file = open(r"C:\Users\Admin\Desktop\NLP\Phase1\NRC Sentiment Lexicon\NRC-Sentiment-Emotion-Lexicons\NRC-Affect-Intensity-Lexicon\NRC-AffectIntensity-Lexicon.txt", "r", encoding="Latin-1")
                for line in nrc_file:
                    line_parts = line.split()
                    test_word = line_parts[0]
                    test_score = int(line_parts[1])
                    test_emotion = line_parts[2]

                    if(word_lowercase == test_word):
                        if(test_emotion == "sadness"):
                            sadness_score = sadness_score + test_score
                            sadness_wordcount = sadness_wordcount + 1

                        if (test_emotion == "joy"):
                            joy_score = joy_score + test_score
                            joy_wordcount = joy_wordcount + 1

                        if (test_emotion == "anger"):
                            anger_score = anger_score + test_score
                            anger_wordcount = anger_wordcount + 1

                        if (test_emotion == "fear"):
                            fear_score = fear_score + test_score
                            fear_wordcount = fear_wordcount + 1

            if(sadness_wordcount != 0):
                sadness_score_f = sadness_score/sadness_wordcount
            else:
                sadness_score_f = -1 #error, defaulting to -1
            if(joy_wordcount != 0):
                joy_score_f = joy_score/joy_wordcount
            else:
                joy_score_f = -1 #error, defaulting to -1
            if(anger_wordcount != 0):
                anger_score_f = anger_score/anger_wordcount
            else:
                anger_score_f = -1 #error, defaulting to -1
            if(fear_wordcount != 0):
                fear_score_f = fear_score/fear_wordcount
            else:
                fear_score_f = -1 #error, defaulting to -1

            sadness_score_list.append(sadness_score_f)
            joy_score_list.append(joy_score_f)
            anger_score_list.append(anger_score_f)
            fear_score_list.append(fear_score_f)

            print("Logically these are all the same: ")
            print(sadness_wordcount)
            print(joy_wordcount)
            print(anger_wordcount)
            print(fear_wordcount)
            nrc_file.close()

word_count_custom_list = []
custom_intensity_score_list = []
def custom_intensity_processing():
    lexicon_file = open(
        r"C:\Users\Admin\Desktop\NLP\Phase1\SO-Cal Lexicon\lexicon_working.txt",
        "r", encoding="Latin-1")
    entry_counter = 0
    for category in blurb_and_descr:
        for entry in category:
            word_count_custom = 0
            score = 0
            words = []
            sentences = sent_tokenize(entry)
            for sentence in sentences:
                words_temp = word_tokenize(sentence)
                for word in words_temp:
                    punctuation_flag = 0
                    for mark in string.punctuation:
                        if mark in word:
                            punctuation_flag = 1
                            break
                    if(punctuation_flag == 0):
                        word_count_custom = word_count_custom + 1
                        word_lowercase = word.lower()
                        words.append(word_lowercase)
            i=0 # keeps track of which word we're on
            for word1 in words: #named word1 jsut to avoid confusion with "word" above
                lexicon_file.seek(0)
                for line in lexicon_file:
                    line_parts = line.split()
                    test_word = line_parts[0]
                    test_score = int(line_parts[1])

                    if ( (i>=3) and (( words[i - 3] + "_" + words[i-2] + "_" + words[i-1] + "_" + words[i] ) == test_word) ):
                        score = score + test_score
                        print("Debug: Lang. Inten. Match found - 4")
                    elif ( (i>=2) and ( ( words[i - 2] + "_" + words[i-1] + "_" + words[i] ) == test_word) ):
                        score = score + test_score
                        print("Debug: Lang. Inten. Match found - 3")
                    elif ((i>=1) and ( (words[i - 1] + "_" + words[i]) == test_word) ):
                        score = score + test_score
                        print("Debug: Lang. Inten. Match found - 2")
                    elif (words[i] == test_word):
                        score = score + test_score
                        print("Debug: Lang. Inten. Match found - 1")
                i = i + 1
            custom_intensity_score_list.append(score)
            word_count_custom_list.append(word_count_custom)
            print("appended: ", score, ", to custom_intensity_score_list... iteration: ", entry_counter)
            entry_counter = entry_counter + 1
    lexicon_file.close()

nrc_score_anger = []
nrc_score_anticipation = []
nrc_score_disgust = []
nrc_score_fear = []
nrc_score_joy = []
nrc_score_negative = []
nrc_score_positive = []
nrc_score_sadness = []
nrc_score_surprise = []
nrc_score_trust = []
file_lines = []
file_words_only = np.empty([150000], dtype="str")
def nrc_emotion_association_processing():
    lexicon_file = open(
        r"C:\Users\Admin\Desktop\NLP\Phase1\NRC Sentiment Lexicon\NRC-Sentiment-Emotion-Lexicons\NRC-Emotion-Lexicon-v0.92\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "r", encoding="Latin-1")
    k=0
    for line in lexicon_file:
        file_lines.append(line)
        line_parts = line.split()
        file_words_only[k] = line_parts[0]
        k = k + 1

    entry_counter = 0
    for category in blurb_and_descr:
        for entry in category:

            nrc_temp_score_anger = 0
            nrc_temp_score_anticipation = 0
            nrc_temp_score_disgust = 0
            nrc_temp_score_fear = 0
            nrc_temp_score_joy = 0
            nrc_temp_score_negative = 0
            nrc_temp_score_positive = 0
            nrc_temp_score_sadness = 0
            nrc_temp_score_surprise = 0
            nrc_temp_score_trust = 0

            words = []
            sentences = sent_tokenize(entry)
            for sentence in sentences:
                words_temp = word_tokenize(sentence)
                for word in words_temp:
                    punctuation_flag = 0
                    for mark in string.punctuation:
                        if mark in word:
                            punctuation_flag = 1
                            break
                    if(punctuation_flag == 0):
                        word_lowercase = word.lower()
                        words.append(word_lowercase)

            i=0 # keeps track of which word we're on
            for word1 in words: #named word1 jsut to avoid confusion with "word" above
                j=0
                theword = words[i]
                for placeholder9 in range(len(file_lines)):
                    if(theword == file_words_only[j]):
                        line_parts = file_lines[j].split()
                        test_word = line_parts[0]
                        test_emotion = line_parts[1]
                        test_score = int(line_parts[2])
                        if(test_score>0):
                            if(test_emotion == "anger"):
                                nrc_temp_score_anger = nrc_temp_score_anger + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "anticipation"):
                                nrc_temp_score_anticipation = nrc_temp_score_anticipation + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "disgust"):
                                nrc_temp_score_disgust = nrc_temp_score_disgust + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "fear"):
                                nrc_temp_score_fear = nrc_temp_score_fear + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "joy"):
                                nrc_temp_score_joy = nrc_temp_score_joy + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "negative"):
                                nrc_temp_score_negative = nrc_temp_score_negative + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "positive"):
                                nrc_temp_score_positive = nrc_temp_score_positive + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "sadness"):
                                nrc_temp_score_sadness = nrc_temp_score_sadness + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "surprise"):
                                nrc_temp_score_surprise = nrc_temp_score_surprise + 1
                                #print("Debug: NRC Emotion match found - 1")
                            if (test_emotion == "trust"):
                                nrc_temp_score_trust = nrc_temp_score_trust + 1
                                #print("Debug: NRC Emotion match found - 1")
                    j = j + 1
                i = i + 1

            nrc_score_anger.append(nrc_temp_score_anger)
            nrc_score_anticipation.append(nrc_temp_score_anticipation)
            nrc_score_disgust.append(nrc_temp_score_disgust)
            nrc_score_fear.append(nrc_temp_score_fear)
            nrc_score_joy.append(nrc_temp_score_joy)
            nrc_score_negative.append(nrc_temp_score_negative)
            nrc_score_positive.append(nrc_temp_score_positive)
            nrc_score_sadness.append(nrc_temp_score_sadness)
            nrc_score_surprise.append(nrc_temp_score_surprise)
            nrc_score_trust.append(nrc_temp_score_trust)
            print("appended entry to nrc_score_lists... iteration: ", entry_counter)
            entry_counter = entry_counter + 1
    lexicon_file.close()


categories = []
with open ('comboC2working.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    for row in readCSV:
        flag = 0
        c = row[2]
        for cat in categories:
            if (cat == c):
                flag = 1
        #Material
        #if(flag==0 and ( c=="Robots" or c=="Footwear" or c=="Woodworking" or c=="Product Design" or c=="Glass" or c=="Wearables" or c=="Fabrication Tools" or c=="Stationery" or c=="Ceramics" or c=="Gadgets" or c=="Camera Equipment")):
        #Experiential
        if(flag==0 and ( c=="Comic Books" or c=="Live Games" or c=="Music Videos" or c=="Thrillers" or c=="Horror" or c=="Documentary" or c=="Performances" or c=="Fiction")):
        #All
        #if (flag == 0 and (1)):
        #Test
        #if (flag == 0 and (c=="Robots" or c=="Glass")):
            categories.append(c)


csvfileread.close()

idx = [[] for k in range(len(categories))] #index in file (very first line is 0 so first data is at 1)
names = [[] for k in range(len(categories))]
blurbs = [[] for k in range(len(categories))]
descriptions = [[] for k in range(len(categories))]
blurb_and_descr = [[] for k in range(len(categories))]

original_rows = [[] for k in range(len(categories))]
date_start = []
date_end = []
total = 0
i=0 #idx
with open ('comboC2working.csv','r', encoding="Latin-1") as csvfileread:
    readCSV = csv.reader(csvfileread, delimiter=',')
    trash = next(readCSV) #takes in the headers in excel so that they don't get included with data
    for row in readCSV:
        l=0 #counts category number
        for cat in categories:
            c = row[2] #categories are column 2 in csv file
            if (c==categories[l]):
                idx[l].append(i)
                names[l].append(row[22])
                blurbs[l].append(row[1])
                descriptions[l].append(row[43])
                date_start.append(row[20])
                date_end.append(row[11])
                total=total+1

                #row_copy = row.copy()
                #row_copy[43] = '"' + row_copy[43] + '"'
                #original_rows[l].append(row_copy)
                original_rows[l].append(row)

            l=l+1
        i = i+1

csvfileread.close()


l=0
for cat in categories:
    i=0
    while i < len(blurbs[l]):
        blurb_and_descr[l].append(blurbs[l][i] + ". " + descriptions[l][i])
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
vader_polarity_list = []
word_count_list = []
analyzer = SentimentIntensityAnalyzer()

smog_word_count_list = []
smog_sentence_count_list = []
nltk_sentence_count_list = []
smog_readability = []
for category in blurb_and_descr:
    i = 0
    for placeholder in category:
        tokenized_description = sent_tokenize(category[i])
        nltk_sentence_count_list.append(len(tokenized_description))
        smog_sentence_count_list.append(textstat.sentence_count(category[i]))


        k = 0
        total_intensity_score = 0
        word_count = 0

        for sentence in tokenized_description:


            ##word counter###
            num_of_words = len(word_tokenize(sentence))
            word_count = word_count + num_of_words
            k=k+1

        word_count_list.append(word_count)

        k=0
        for sentence in tokenized_description:
            num_of_words_sentence = len(word_tokenize(sentence))
            result = analyzer.polarity_scores(sentence)
            print(result)  # just debugging, can delete
            intensity_score = result['compound']
            print("Sentence ", k, "intensity score: ", intensity_score)
            #total_intensity_score = total_intensity_score + intensity_score*(num_of_words_sentence/word_count)
            total_intensity_score = total_intensity_score + intensity_score
            k = k + 1

        if (k):
            print("total sentences: ", k)
            print("total intensity score: ", total_intensity_score)
            vader_polarity_list.append((total_intensity_score/k))
        else:
            print("vader - Error! No description! (0 sentences counted)")
            vader_polarity_list.append(9)  # score of 9 for empty descriptions

        smog_word_count_list.append(textstat.lexicon_count(category[i], removepunct=True))
        smog_readability.append(textstat.smog_index(category[i]))
        i=i+1

##### end vader stuff ####

### this part isn't necessary but puts idx into a single layer row major list ####
final_idx_list = []
for placeholder5 in idx:
    for placeholder6 in placeholder5:
        final_idx_list.append(placeholder6)

### end this part isn't necessary but puts idx into a single layer row major list ####

textblob_implemenation()




normalization_and_graphing() #calls normalization and graphing, not necessary, but remember to remove from final csv file write too.

simple_valence_calculator()

duration_calculation()



#### Writing and finalizing everything#####

#debugging lengths to match, can delete ###

print(smog_readability)
print(vader_polarity_list)
print(textblob_polarity_list)
print(textblob_subjectivity_list)
print(logged_list_vader)
print(logged_list_textblob_polarity)
print(logged_list_textblob_subjectivity)
print(everything_included_list_vader)
print(everything_included_list_textblob_polarity)
print(everything_included_list_textblob_subjectivity)
print(everything_included_outlier_list)
print(valence_list)
print(duration_list)


print("Testing final lengths these shoulda all match: ")

print(len(smog_readability))
print(len(vader_polarity_list))
print(len(textblob_polarity_list))
print(len(textblob_subjectivity_list))
print(len(logged_list_vader))
print(len(logged_list_textblob_polarity))
print(len(logged_list_textblob_subjectivity))
print(len(everything_included_list_vader))
print(len(everything_included_list_textblob_polarity))
print(len(everything_included_list_textblob_subjectivity))
print(len(everything_included_outlier_list))
print(len(valence_list))
print(len(duration_list))
###############################################


#nrc_intensity_processing()

custom_intensity_processing()

nrc_emotion_association_processing()


####### writing ###########
rows=[]
row=[]
l=0
overall_tracker = 0 #tracks which logged one we're on bc outliers are removed
for placeholder1 in idx:
    i = 0
    for placeholder2 in placeholder1:
        #all available data lists to print#
        row = [categories[l], idx[l][i], names[l][i], blurbs[l][i], descriptions[l][i], blurb_and_descr[l][i], word_count_list[overall_tracker], smog_word_count_list[overall_tracker], smog_readability[overall_tracker], vader_polarity_list[overall_tracker], abs(vader_polarity_list[overall_tracker]), textblob_polarity_list[overall_tracker], textblob_subjectivity_list[overall_tracker], logged_list_vader[overall_tracker], logged_list_textblob_polarity[overall_tracker], logged_list_textblob_subjectivity[overall_tracker], everything_included_list_vader[overall_tracker], everything_included_list_textblob_polarity[overall_tracker], everything_included_list_textblob_subjectivity[overall_tracker], everything_included_outlier_list[overall_tracker], valence_list[overall_tracker], duration_list[overall_tracker], 9, 9, 9, 9, custom_intensity_score_list[overall_tracker], word_count_custom_list[overall_tracker], nltk_sentence_count_list[overall_tracker], smog_sentence_count_list[overall_tracker], nrc_score_anger[overall_tracker], nrc_score_anticipation[overall_tracker], nrc_score_disgust[overall_tracker], nrc_score_fear[overall_tracker], nrc_score_joy[overall_tracker], nrc_score_negative[overall_tracker], nrc_score_positive[overall_tracker], nrc_score_sadness[overall_tracker], nrc_score_surprise[overall_tracker], nrc_score_trust[overall_tracker] ]
        #row = [idx[l][i], blurb_and_descr[l][i], word_count_list[overall_tracker], smog_word_count_list[overall_tracker], word_count_custom_list[overall_tracker], nltk_sentence_count_list[overall_tracker], smog_sentence_count_list[overall_tracker], smog_readability[overall_tracker], vader_polarity_list[overall_tracker], abs(vader_polarity_list[overall_tracker]), textblob_polarity_list[overall_tracker], textblob_subjectivity_list[overall_tracker], everything_included_outlier_list[overall_tracker], valence_list[overall_tracker], duration_list[overall_tracker], custom_intensity_score_list[overall_tracker], nrc_score_anger[overall_tracker], nrc_score_anticipation[overall_tracker], nrc_score_disgust[overall_tracker], nrc_score_fear[overall_tracker], nrc_score_joy[overall_tracker], nrc_score_negative[overall_tracker], nrc_score_positive[overall_tracker], nrc_score_sadness[overall_tracker], nrc_score_surprise[overall_tracker], nrc_score_trust[overall_tracker] ]
        rows.append(list(row))

        overall_tracker = overall_tracker + 1
        i=i+1
    l=l+1

l=0
final_original_rows =[]
for placeholder3 in original_rows:
    i=0
    for placeholder4 in placeholder3:
        final_original_rows.append(placeholder4)
        i=i+1
    l=l+1


k = 0
with open ('init1_output2.csv','w', encoding="Latin-1") as csvfilewrite:
    writeCSV = csv.writer(csvfilewrite, delimiter=',', lineterminator='\n')
    #all available data lists to print#
    writeCSV.writerow(["backers_count", "blurb", "category", "converted_pledged_amount", "country", "created_at",  "creator", "currency", "currency_symbol", "currency_trailing_code", "current_currency", "deadline", "disable_communication", "friends", "fx_rate", "goal",	"id", "is_backing", "is_starrable", "is_starred", "launched_at", "location", "name", "permissions	", "photo",	"pledged", "profile", "slug", "source_url", "spotlight", "staff_pick", "state", "state_changed_at", "static_usd_rate", "urls", "usd_pledged", "usd_type", "projects_failed", "projects_canceled", "projects_suspended", "projects_successful", "projects_total", "projects_success_ratio", "description", "comment_count",	"updates_count", "faq_count", "categories", "idx (index of appearance in original, cleaned file)", "name", "blurb", "description", "blurb_and_descr_cleaned", "NLTK_word_count_list", "smog_word_count_list", "smog_readability", "vader_polarity_score_raw","vader_polarity_score_absolute_value", "textblob_polarity_score_raw", "textblob_subjectivity_score_raw", "vader_polarity_score_normalized", "textblob_polarity_score_normalized", "textblob_subjectivity_score_normalized", "vader_polarity_score_normalized_all_included", "textblob_polarity_score_normalized_all_included", "textblob_subjectivity_score_normalized_all_included", "outlier_status", "valence", "duration_days", "sadness_score", "joy_score", "anger_score", "fear_score", "language_intensity", "NLTK_word_count_without_punctuation", "NLTK_sentence_count", "SMOG_sentence_count", "nrc_score_anger", "nrc_score_anticipation", "nrc_score_disgust", "nrc_score_fear", "nrc_score_joy", "nrc_score_negative", "nrc_score_positive", "nrc_score_sadness", "nrc_score_surprise", "nrc_score_trust"])
    #writeCSV.writerow(["backers_count", "blurb", "category", "converted_pledged_amount", "country", "created_at",  "creator", "currency", "currency_symbol", "currency_trailing_code", "current_currency", "deadline", "disable_communication", "friends", "fx_rate", "goal",	"id", "is_backing", "is_starrable", "is_starred", "launched_at", "location", "name", "permissions	", "photo",	"pledged", "profile", "slug", "source_url", "spotlight", "staff_pick", "state", "state_changed_at", "static_usd_rate", "urls", "usd_pledged", "usd_type", "projects_failed", "projects_canceled", "projects_suspended", "projects_successful", "projects_total", "projects_success_ratio", "description", "comment_count",	"updates_count", "faq_count", "idx (index of appearance in original, cleaned file)", "blurb_and_descr_cleaned", "NLTK_word_count_list", "smog_word_count_list", "NLTK_word_count_without_punctuation", "NLTK_sentence_count", "SMOG_sentence_count", "smog_readability", "vader_polarity_score_raw","vader_polarity_score_absolute_value", "textblob_polarity_score_raw", "textblob_subjectivity_score_raw", "outlier_status", "valence", "duration_days", "language_intensity", "nrc_score_anger", "nrc_score_anticipation", "nrc_score_disgust", "nrc_score_fear", "nrc_score_joy", "nrc_score_negative", "nrc_score_positive", "nrc_score_sadness", "nrc_score_surprise", "nrc_score_trust"])

    for r in rows:
        final_row = final_original_rows[k] + r
        writeCSV.writerow(final_row)
        k=k+1
csvfilewrite.close()

print("Random final list length debug these should all be same: ")
print(len(custom_intensity_score_list))
print(len(word_count_list))
print(len(final_idx_list))
print(len(textblob_subjectivity_list))
print(len(valence_list))
print(len(duration_list))
print(len(nrc_score_anger))
print(len(nrc_score_negative))
print(len(nrc_score_trust))

######## end writing ###########