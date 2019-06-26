#Author: Chetan Singh

#Importing Libraries
import pandas as pd
import numpy as np

#Import the cleaned dataset
data_raw = pd.read_csv('final_cleaned.csv', header = None)

#Create set of labels
label_lst = []
for x in data_raw[1]:
    for label in x.split(', '):
        label_lst.append(label)

#Form a unique set of labels
label_set = set(label_lst)

#Form all the labels as columns
main_df = pd.DataFrame(columns = list(label_set), data = [np.zeros(len(label_set), dtype = int) for x in range(len(data_raw))])

main_df.insert(0, 'word_tokens', data_raw[0])

#Set values as 1 where a document belongs to that label
for row in range(len(data_raw)):
    labels = data_raw[1][row].split(', ')
    for l in range(len(labels)):
        main_df.at[row, labels[l]] = 1

#Save the dataframe to a .csv file
main_df.to_csv('sparse_matrix.csv', index = False)