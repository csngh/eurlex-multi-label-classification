#Author: Chetan Singh

#Import libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import platform

#Import the csv containing scraped text and their lables
df = pd.read_csv('final_scraped.csv', header = None)

#Initialize Stopword corpus and Stemmer function
sw_corpus = stopwords.words('english')
stemmer = PorterStemmer()

#Run stopword removal and stemming over all texts containging in the data frame
for row_count in range(len(df)):
    #Print progress %
    if (platform.system().lower() == 'windows'):
    	os.system('cls')
    else:
    	os.system("printf '\\033c'")
    print("Processing... Stopwords Removal/Stemming/Cleaning...\nProgress: {:.2f}".format(row_count/len(df)*100), '%')
    
    #Only selecting those rows which has text
    if 'str' in str(type(df.at[row_count, 0])):
        para_word_tokens = df.at[row_count, 0].split()
        #Removing stopwords and stemming
        sw_rem_tokens = [word for word in para_word_tokens if word not in sw_corpus]
        stemmed_tokens = [stemmer.stem(word) for word in sw_rem_tokens]
        #Removing digits and words which ha less than 3 chars
        cleaned_tokens_0 = [word for word in stemmed_tokens if not any(i.isdigit() for i in word)]
        cleaned_tokens_1 = [word for word in cleaned_tokens_0 if len(word) > 2]
        #Removing unnecessary symbols
        for count, word in enumerate(cleaned_tokens_1):
            x = word
            for i in word:
                if not i.isalpha():
                    x = x.replace(i, '')
            cleaned_tokens_1[count] = x
        
        #Make changes
        df.at[row_count, 0] = ' '.join(cleaned_tokens_1)

#Cleaning the labels
for row in range(len(df)):
    labels = df.at[row, 1].split(', ')
    for label_index in range(len(labels)):
        x = labels[label_index]
        for i in labels[label_index]:
            if not i.isalpha() and i != '_':
                if i == '-':
                    x = x.replace(i, '_')
                else:
                    x = x.replace(i, '')
        labels[label_index] = x
    df.at[row, 1] = ', '.join(labels)

#Write to .csv
df.to_csv('final_cleaned.csv', header=False, index=False)

