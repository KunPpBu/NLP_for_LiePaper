import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import transformers
from transformers import pipeline

pd.set_option("display.max_colwidth", 200)

document = pd.read_csv("ca_data.csv", usecols=['text'], lineterminator='\n')
news_df = pd.DataFrame(document)
news_df = news_df.astype(str)
# removing everything except alphabets`
news_df['clean_doc'] = news_df['text'].str.replace("[^a-zA-Z#]", " ")

# removing hashtag
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: re.sub("#[A-Za-z0-9_]+", "", x))

# removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc

# calling the pipeline
summarizer = pipeline("summarization")
news_df['summary']=news_df['clean_doc'].astype('str')
news_df['summary']= pd.Series(news_df['summary'], dtype = 'string')
#news_df['summary']=news_df['summary'].apply(lambda x : summarizer(x, min_length=1, max_length=10))
summary_df =  []
for i in range(0, len(news_df)):
    res = summarizer(news_df['summary'][i], min_length=1, max_length=10)
    summary_df.append(res)

summary_df = pd.DataFrame(summary_df, index=range(0,len(summary_df)))
test = summary_df[0]
test_list=[]
for i in range(0, len(test)):
    res = list(test[i].values())
    test_list.append(res)
flattened_list = [val for sublist in test_list for val in sublist]
flattened_list
summary_list = flattened_list
news_df['summary'] = summary_list


from nltk.sentiment import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
def find_sentiment(post):
    try:
        if sia.polarity_scores(post)["compound"] > 0:
            return "Positive"
        elif sia.polarity_scores(post)["compound"] < 0:
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"


def plot_sentiment(df, feature, title):
    counts = df[feature].value_counts()
    percent = counts / sum(counts)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    colors = ["green", "red", "blue"]
    counts.plot(kind='bar', ax=ax1, color=colors)
    percent.plot(kind='bar', ax=ax2, color=colors)
    ax1.set_ylabel(f'Counts : {title} sentiments', size=12)
    ax2.set_ylabel(f'Percentage : {title} sentiments', size=12)
    plt.suptitle(f"Sentiment analysis: {title}")
    plt.tight_layout()
    plt.show()

news_df['summary_text_sentiment'] = news_df['summary'].apply(lambda x: find_sentiment(x))
plot_sentiment(news_df, 'summary_text_sentiment', 'Summarized Text')



text_sentiment = news_df['summary_text_sentiment']
news_df.to_csv('sentiment_output_summary_text.csv', index=True)



import transformers
from transformers import pipeline
summary = news_df['summary'].astype(str)
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_id)
result = sentiment_pipeline(summary)
result_df = pd.DataFrame(result, index=range(1, len(result)+1))