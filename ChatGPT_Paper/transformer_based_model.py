import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import seaborn as sns
import re
import nltk
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

#clean_list = news_df['clean_doc'].values.tolist()


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

news_df['text_sentiment'] = news_df['clean_doc'].apply(lambda x: find_sentiment(x))
plot_sentiment(news_df, 'text_sentiment', 'Full Text')

text_sentiment = news_df['text_sentiment']
news_df.to_csv('sentiment_output_full_text.csv', index=True)

