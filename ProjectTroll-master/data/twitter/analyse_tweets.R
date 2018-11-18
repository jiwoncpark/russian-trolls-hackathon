library(dplyr)
library(tm)
library(qdap)

corpus <- Corpus(VectorSource(read.csv('./mydata/twitter.csv',nrows = 1000)$content))

skipWords <- function(x) removeWords(x, stopwords("english"))
funcs <- list(tolower, removePunctuation, removeNumbers, stripWhitespace, skipWords)
corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = funcs)

