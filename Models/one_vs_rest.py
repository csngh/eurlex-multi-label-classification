#Author: Chetan Singh

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, hamming_loss, jaccard_score, f1_score
from sklearn.multiclass import OneVsRestClassifier

data_raw = pd.read_csv('sparse_matrix.csv')
data_raw.shape

rowSums = data_raw.iloc[:, 1:].sum(axis = 1)
no_labels_count = (rowSums == 0).sum(axis = 0)

print("Total number of docs = ",len(data_raw))
print("Number of docs with no labels = ",no_labels_count)
print("Number of docs with labels = ",(len(data_raw) - no_labels_count))

labels = list(data_raw.columns.values)
labels = labels[1:]
print(f'Total labels: {len(labels)}')

counts = []
for label in labels:
    counts.append((label, data_raw[label].sum()))
df_stats = pd.DataFrame(counts, columns=['label', 'number of docs'])
(df_stats.sort_values(by = ['number of docs'], inplace = True))
df_stats.reset_index(inplace = True)
df_stats.drop(['index'], axis = 1, inplace = True)
df_stats.loc[np.random.choice(df_stats.index, size = 10)]

train, test = train_test_split(data_raw, test_size = 0.30, shuffle = True)

print(train.shape)
print(test.shape)

train_text = train['word_tokens']
test_text = test['word_tokens']

print('Vectorizing...')
vectorizer = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word', ngram_range = (1, 3), norm = 'l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)

print('Transforming...')
x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['word_tokens'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['word_tokens'], axis=1)

print('Building Classifier...')
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])
acc_mean = 0
rec_mean = 0
prec_mean = 0
hm_mean = 0
jc_mean = 0
f1_mean = 0
for count, label in enumerate(labels):
    print('###Count: {}, label: {} ...###'.format(count, label))
    
    LogReg_pipeline.fit(x_train, train[label])
    
    prediction = LogReg_pipeline.predict(x_test)

    acc_mean += accuracy_score(test[label], prediction)
    rec_mean += recall_score(test[label], prediction)
    prec_mean += precision_score(test[label], prediction)
    hm_mean += hamming_loss(test[label], prediction)
    jc_mean += jaccard_score(test[label], prediction)
    f1_mean += f1_score(test[label], prediction, average = 'macro')

    print("\n")

acc_mean = acc_mean / len(labels)
rec_mean = rec_mean / len(labels)
prec_mean = prec_mean / len(labels)
jc_mean = jc_mean / len(labels)
f1_mean = f1_mean / len(labels)
hm_mean = hm_mean / len(labels)

print('Test accuracy {:.2f}'.format(acc_mean))
print('Test recall {:.2f}'.format(rec_mean))
print('Test precision {:.2f}'.format(prec_mean))
print('Test hamming loss {:.2f}'.format(hm_mean))
print('Test jaccard score {:.2f}'.format(jc_mean))
print('Test F1 score macro {:.2f}'.format(f1_mean))
print("\n")

print("\n")
