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
#news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

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

news_df['text_sentiment'] = news_df['clean_doc'].apply(lambda x: find_sentiment(x))
plot_sentiment(news_df, 'text_sentiment', 'Full Text')



news_df['summary_text_sentiment'] = news_df['summary'].apply(lambda x: find_sentiment(x))
plot_sentiment(news_df, 'summary_text_sentiment', 'Summarized Text')



#text_sentiment = news_df['summary_text_sentiment']
news_df.to_csv('sentiment_output_summary_text.csv', index=True)



import transformers
from transformers import pipeline
#summary = pd.Series(news_df['summary'], dtype = 'string')
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_id)
result = news_df['clean_doc'].apply(lambda x: sentiment_pipeline(x))
result_df = pd.DataFrame(result, index=range(1, len(result)+1))
test1 = result_df['clean_doc']
test_list1=[]
for i in range(1, len(test1)):
    res = {k: v for d in test1[i] for k, v in d.items()}
    test_list1.append(res)

test1_df = pd.DataFrame(test_list1)
test1_df = test1_df.rename(columns={"label" : "transformer_sentiment_fulltext", "score": "transformer_score_fulltext"})

compare_df = pd.concat([news_df,test1_df], axis=1, join="inner")
compare_df = compare_df.rename(columns={"text_sentiment" : "nltk_sentiment_fulltext",
                                        "summary_text_sentiment": "nltk_sentiment_summary"})
compare_df.to_csv('compare_results_transformer_vs_nltk.csv', index= False)

compare_senti = compare_df[["nltk_sentiment_fulltext", "nltk_sentiment_summary", "transformer_sentiment_fulltext"]]

plot_sentiment(compare_df, 'transformer_sentiment_fulltext', 'Transformer Text')

#transformer for summarization
result2 = news_df['summary'].apply(lambda x: sentiment_pipeline(x))
result2_df = pd.DataFrame(result2, index=range(1, len(result)+1))
test2 = result2_df['summary']
test_list2=[]
for i in range(1, len(test2)):
    res = {k: v for d in test2[i] for k, v in d.items()}
    test_list2.append(res)

test2_df = pd.DataFrame(test_list2)
test2_df = test2_df.rename(columns={"label" : "transformer_sentiment_summary", "score": "transformer_score_summary"})


compare_df2 = pd.concat([compare_df,test2_df], axis=1, join="inner")
compare_df2.to_csv('compare_results_transformer_vs_nltk.csv', index= False)



p1 = compare_df2["nltk_sentiment_fulltext"].value_counts()\
    .plot(kind='bar',
          title = 'Count of full text tweets by sentiments analysis using nltk',
          figsize= (10,8))
#p1.set_xlabel('Sentiment by nltk for full text')


plt.show()

p2 = compare_df2["nltk_sentiment_summary"].value_counts()\
    .plot(kind='bar',
          title = 'Count of summarized tweets by sentiments analysis using nltk',
          figsize= (10,8))


p3 = compare_df2["transformer_sentiment_fulltext"].value_counts()\
    .plot(kind='bar',
          title = 'Count of full text tweets by sentiments analysis using transformer',
          figsize= (10,8))


p4 = compare_df2["transformer_sentiment_summary"].value_counts()\
    .plot(kind='bar',
          title = 'Count of summarized tweets by sentiments analysis using transformer',
          figsize= (10,8))


# fig, axs = plt.subplot(1, 3, figsize = (12,3))
