# tokens <- tokenize(rc$votes.long$description)
# dictionary <- tokens %>% unlist %>% table %>% sort %>% names
# ranks <- lapply(tokens, match, dictionary, nomatch=0L)
# still needs keras::pad_sequences() or similar since ragged, see https://stackoverflow.com/questions/49594691/how-to-pad-text-sequences-in-r-using-keras-and-pad-sequences

tokenize <- function(txt, x, lang="english", remove_mentions=FALSE, remove_links=TRUE, remove_hashtags=FALSE){
  
  require(tm) # 'text mining' library
  
  langs <- c("danish", "dutch", "english", 
             "finnish", "french", "german", 
             "hungarian", "italian", "norwegian", 
             "portuguese", "russian", "spanish", "swedish")
  
  if(length(txt) == 1){   
    
    keepers <- unlist(strsplit(tolower(txt), " ")) # tokens
    if(remove_mentions)
      keepers <- keepers[!grepl("@", tokens)]
    if(remove_links)
      keepers <- keepers[!grepl("https", keepers)]
    if(remove_hashtags)
      keepers <- keepers[!grepl("#", keepers)]
    keepers <- removePunctuation(keepers)
    keepers <- keepers[nchar(keepers) > 0]
    
    w <- agrep(lang, langs) # approx grep
    
    if(length(w))
      keepers <- setdiff(keepers, stopwords(langs[w]))
    
    if(length(keepers)) return(keepers) else NA
    
  }else{
    
    out <- list()
    
    for(i in 1:length(txt)){
      out[[i]] <- tokenize(txt[i], x, lang[i])
      if(i %% 501 == 0) cat(".")
    }
    cat("\n")  
    
    return(out) 
  }
}

# process tokens into ranks and/or possibly padded sequences
process_tokens <- function(tokens, 
                           return_type = c("padded_sequences", # default return option is padded sequences
                                                   "ranks", "dictionary", "all"), 
                           pad_seq_max_len = 10){ # padded sequence max length may be ignored
  
  return_type <- match.arg(return_type, c("padded_sequences", "ranks", "dictionary", "all"))
  
  dictionary <- names(sort(table(unlist(tokens))))
  
  if(return_type != "dictionary"){
    
    ranks <- lapply(tokens, match, dictionary, nomatch=0L)
    
    if(return_type %in% c("padded_sequences", "all")){
      
      N <- length(tokens)
      P <- pad_seq_max_len
      padded <- matrix(0, nrow = N, ncol = P)
      for(i in 1:N){
        K <- min(P, length(ranks[[i]]))
        padded[i, 1:K] <- ranks[[i]][1:K]
        if(i %% 501 == 0) cat(".")
      }
      cat("\n")
      if(return_type == "padded_sequences"){
        return(padded)
      }else{
        return(list(dictionary=dictionary, padded=padded, ranks=ranks))
      }
    }else{
      return(ranks)
    }
    
  }else{
    return(dictionary)
  }
}

