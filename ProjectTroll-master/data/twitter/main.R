source("data/twitter/get_troll_data.R")
source("data/twitter/txt_helper_functions.R")
library(dplyr)
library(tm) # Text Mining, https://www.rdocumentation.org/packages/tm/versions/0.7-5

# all data
#troll_tweets <- get_troll_data(data_dir = "mydata")

# small chunk of 2000 per file
troll_tweets <- get_troll_data(data_dir = "mydata", nrows=2000)

# filter non enlgish tweets
troll_tweets <- troll_tweets %>% dplyr::filter(language == "English")

# keep only the content and the publish date
out <- troll_tweets %>% dplyr::select(content, publish_date)

# write to csv
write.csv(out, file="/home/jwp/stage/stats285-experiment-management-system/hackathon/ProjectTroll-master/mydata/twitter.csv",  row.names=FALSE)

