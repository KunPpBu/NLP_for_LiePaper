data <- read.csv("tweets.csv")
head(data)

location=data$user_location
count_location=table(location)
pdf(file="Users_Location.pdf",,width=8,height=8)
pie(count_location, main="Occurrence by users location",cex=0.45,col=gray(seq(0.4, 1.0, length = 6)))
dev.off()

#Verfired users
data1<- data[which(data$user_verified=="True"),]

#CA users
state <- "\\bCA\\b"
tmp <- toupper(data1$user_location)
id=grep(state,tmp)
ca_data <- data1[id,]

ca_location0=ca_data$user_location
id <- grep(":", ca_location0)
ca_location01 <- ca_location0[-id] 
id1 <- grep("∙", ca_location01)
ca_location <- ca_location01[-id1]
count_ca.location=table(ca_location)


pdf(file="CA_Users_Location.pdf",,width=8,height=8)
pie(count_ca.location, main="ChatGPT tweets by verified twitter users location within California",cex=0.45,col=gray(seq(0.4, 1.0, length = 6)))
dev.off()

# tweets post date
date1 = NULL
date0 <- data$date
for(i in 1: length(date0)){
  date1[i] <- strsplit(date0, " ")[[i]][1]
}
head(unlist(date1))
date1 <- unlist(date1)
date1 <- data.frame(date1)
date1$Date <- as.Date(date1$date1)
library(ggplot2)
library(scales)

ggplot(date1, aes(x=Date)) + geom_histogram(binwidth=5, colour="white") +
  scale_x_date(labels = date_format("%Y-%m-%d"),
               breaks = seq(min(date1$Date)-30, max(date1$Date)+30, 5),
               limits = c(as.Date("2022-12-01"), as.Date("2023-01-10"))) +
  ylab("Frequency") + xlab("Date") +
  theme_bw() 


install.packages("wordcloud")
library(wordcloud)
install.packages("RColorBrewer")
library(RColorBrewer)
install.packages("wordcloud2")
library(wordcloud2)

install.packages("tm")
library(tm)
#Create a vector containing only the text
text <- ca_data$text
# Create a corpus  
docs <- Corpus(VectorSource(text))

gsub("https\\S*", "", ca_data$text) 
gsub("@\\S*", "", ca_data$text) 
gsub("amp", "", ca_data$text) 
gsub("[\r\n]", "", ca_data$text)
gsub("[[:punct:]]", "", ca_data$text)

install.packages("magrittr") # package installations are only needed the first time you use it
install.packages("dplyr")    # alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%

docs <- docs %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))

dtm <- TermDocumentMatrix(docs) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)

install.packages("tidytext")
library(tidytext)
tweets_words <-  ca_data %>%
  select(text) %>%
  unnest_tokens(word, text)
words <- tweets_words %>% count(word, sort=TRUE)

set.seed(1234) # for reproducibility 
wordcloud(words = df$word, freq = df$freq, min.freq = 1, 
          max.words=300, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

