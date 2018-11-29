Toll of the Trolls: Exploratory Data Analysis
================
Pete Mohanty
10/29/2018

This assignment is designed to familiarize yourself with the data which we will use for the hack-a-thon. It is intended to be completed in `R`, in which it can be completed with a very small number of lines of code but of course feel free to use a language of your choice. You do not need to complete the assignment using the entire data set; a subset is recommended.

Getting Started
===============

Download the R scripts and adjust the paths as needed. If the `nrows=2000` is deleted, you will download the entire dataset.

``` r
source("../data/twitter/get_troll_data.R")
source("../data/twitter/txt_helper_functions.R")
library(dplyr)
library(tm) # Text Mining, https://www.rdocumentation.org/packages/tm/versions/0.7-5
library(ggplot2)

# all data into (optional) subdirectory
# troll_tweets <- get_troll_data(data_dir="mydata")

# small chunk of 2000 per file into current working data
troll_tweets <- get_troll_data(nrows=2000, data_dir = "../data") %>% filter(language == "English")
```

**Q1** Which variables are included in the `data.frame`? What is the unit of observation for `followers` and `following`? What do you find surprising or noteworthy about the data?

**Q2** Make a few notes on each of the account categories. Next, read through the tweets and display a few representative ones for each type.

**Q3** Briefly comment as to the relationship between the number of followers, followed, and updates. Also, do the numeric variables appear normally distributed? Does the relationship differ if the data are aggregated by author?

**Q4** Plot one or more of the quantitative variables over time, either coloring or faceting by label (e.g., account category). What trends do you notice?

**Q5** Familiarize yourself with <https://www.tidytextmining.com/>. For each of the categories, tokenize the content of each tweet and plot a barplot of most common words (which are not stopwords or punctuation). What trends do you notice?

**Q6** Following Chapter 2 of <https://www.tidytextmining.com/>, perform a sentiment analysis, again subsetting by account category. What patterns or differences in sentiment do you notice?

**Q7** Following Chapter 6 of <https://www.tidytextmining.com/>, fit topic models (after filtering by account category). Based on your results from **Q6**, how many different topics do you think will be necessary for each? What are your results?
