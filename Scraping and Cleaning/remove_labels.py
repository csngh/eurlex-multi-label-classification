#Author: Chetan Singh

#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

#Read the cleaned data from .csv as dataframe
df = pd.read_csv('final_cleaned.csv', header = None)

#Store all labels in a list
label_lst = []
for x in df[1]:
    for label in x.split(', '):
        label_lst.append(label)

#Find unique labels
label_set = set(label_lst)

#Count all occurences for the unique labels
occurence_list = []
for label in label_set:
    count = 0
    for x in label_lst:
        if x == label:
            count += 1
    occurence_list.append([label, count])

#Find all labels with less occurences
cls_with_less_count = []
for i in range(1, 8):
    cls_with_less_count.extend([x[0] for x in occurence_list if x[1] == i])

#Remove all the labels from the dataset
for i in range(len(df[1])):
    x = df.at[i,1].split(', ')
    for label in df.at[i,1].split(', '):
        if label in cls_with_less_count:
            x.remove(label)
    df.at[i, 1] = ', '.join(x)

#Remove contents with no labels
to_rem = []
for i in range(len(df)):
    for label in df.at[i, 1].split(', '):
        if label == '':
            to_rem.append(i)

#Modify and save final output to .csv
df.drop(to_rem, inplace = True)
df.reset_index(inplace=True)
df.drop('index', inplace=True, axis=1)

df.to_csv('imbalanced_labeled_removed.csv', header = False, index = False)


