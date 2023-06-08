import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("compare_results_transformer_vs_nltk.csv")


# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label

pred_nltk = le.fit_transform(df['transformer_sentiment_summary'])
label_nltk = le.fit_transform(df['transformer_sentiment_fulltext'])

pred_nltk[pred_nltk == 1] =2
label_nltk[label_nltk ==1] =2


#print("Accuracy", metrics.accuracy_score(label_nltk, pred_nltk))
acc = metrics.accuracy_score(label_nltk, pred_nltk)
fpr, tpr, thresh = metrics.roc_curve(label_nltk, pred_nltk,pos_label=2)
auc = metrics.roc_auc_score(label_nltk, pred_nltk)
plt.plot(fpr,tpr,label="Transformer, auc="+str(round(auc,3)))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Sentiment analysis AUC with Transformer \n accuracy = '+str(round(acc,3)))
plt.show()