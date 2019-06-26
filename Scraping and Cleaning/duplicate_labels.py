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
    cls_with_less_count.append([x[0] for x in occurence_list if x[1] == i])
    
#Make duplicates of instances with less label occurences
chosen_labels = []
duplicated_instances = []
for i in range(7):
    clear_output(wait = True)
    print('Working on range ', i, ' ...')
    for label in cls_with_less_count[i]:
        for row in range(len(df[1])):
            for df_label in df[1][row].split(', '):
                if label == df_label:
                    if label not in chosen_labels:
                        for count in range(1, 8 - i):
                            duplicated_instances.append([df[0][row], label])
                    chosen_labels.append(label)
                    
#Attach the duplicates with a dataframe
df_duplicates = pd.DataFrame(duplicated_instances)
frames = [df, df_duplicates]
final_df = pd.concat(frames)

#Save the final output in .csv file
final_df.reset_index(inplace=True)
final_df.drop('index', inplace=True, axis=1)

final_df.to_csv('imbalanced_labeled_duplicated.csv', header = False, index = False)


