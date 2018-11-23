# returns all five twitter Russian troll data sets as one data.frame.
# checks working directory (or directory provided) for data. 
# downloads if need be to data_dir.
# ... represents additional parameters to be passed to read.csv(), such as nrows=25000.
#     note stringsAsFactors is set to FALSE and should remain so.
# examples:
# troll_tweets <- download_or_read_csv()
# troll_tweets <- download_or_read_csv("my_data_folder")
# troll_tweets <- download_or_read_csv("data", nrows=25000)
get_troll_data <- function(..., data_dir = NULL){
  
  data_dir <- if(is.null(data_dir)) getwd() else data_dir
  
  troll_list <- list()
  cat("starting download ...\n\n")
  for(i in 1:13){
    
    file_name <- paste0("troll_tweets", i, ".csv")
    link <- paste0("https://raw.githubusercontent.com/fivethirtyeight/russian-troll-tweets/master/IRAhandle_tweets_",
                    i, ".csv") 
    
    
    if(file_name %in% dir(data_dir)){
      
      troll_list[[i]] <- read.csv(file.path(data_dir, file_name), 
                                  stringsAsFactors = FALSE, ...)
      cat("loaded file", i, "...\n")
      
    }else{
      tried <- try(troll_list[[i]] <- read.csv(link, 
                                  stringsAsFactors = FALSE, ...))
      if(inherits(tried, "try-error")){
        warning("unable to download file:", i, "\n\n")
      }else{
        
        cat("downloaded file ", i,  "... saving copy to disk... ")
        
        if(!(data_dir %in% dir()))
           stop("As a precaution, please create the directory you wish to download the data to before calling this function.")
        
        write.csv(troll_list[[i]],
                  file = file.path(data_dir, file_name), 
                  row.names = FALSE)
        cat("done...\n")
      }
    }
  }
  
  troll_df <- do.call(rbind, troll_list)
  rm(troll_list)
  return(troll_df)
  
}

