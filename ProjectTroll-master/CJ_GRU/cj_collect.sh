#!/bin/bash -l

#READ remaining_list.cjr and FIND The counters that need
#to be collected
    
if [ ! -s completed_list.cjr ]; then
    
    if [ ! -s  remaining_list.cjr ]; then
     echo "CJ::Reduce:: All results completed and collected. ";
    else
    # check if collect is complete
    # if yes, then echo results collect fully
    echo "     CJ::Reduce:: Nothing to collect. Possible reasons are: Invalid filename, No new completed job.";
    fi
    
else
  
    TOTAL=$(wc -l < "completed_list.cjr");
    
    # determine whether reduce has been run before
    if [ ! -f "results/training_results.csv" ];then
      # It is the first time reduce is being called.
      # Read the result of the first package
    
      firstline=$(head -n 1 completed_list.cjr)
      COUNTER=`grep -o "[0-9]*" <<< $firstline`

      mkdir -p "$(dirname "results/training_results.csv")" && touch "results/training_results.csv"
    
      cat "$COUNTER/results/training_results.csv" > "results/training_results.csv";
    
        # Pop the first line of remaining_list and add it to collect_list
    #  sed -i '1d' completed_list.cjr
        if [ ! -f collect_list.cjr ];then
            echo $COUNTER > collect_list.cjr;
        else
          echo "CJ::Reduce:: CJ in AWE. collect_list.cjr exists but CJ thinks its the first time reduce is called" 1>&2
          exit 1
        fi
    PROGRESS=1;
    percent_done=$(awk "BEGIN {printf \"%.2f\",100*${PROGRESS}/${TOTAL}}")
    printf "\n SubPackage %d Collected (%3.2f%%)" $COUNTER $percent_done

    else
    PROGRESS=0;
    fi

    
    for LINE in $(tail -n +$(($PROGRESS+1)) completed_list.cjr);do


        PROGRESS=$(($PROGRESS+1))
        # Reduce results
        COUNTER=`grep -o "[0-9]*" <<< $LINE`


        # Remove header-lines!
        startline=$((1+1));
        sed -n "$startline,\$p" < "$COUNTER/results/training_results.csv" >> "results/training_results.csv";  #simply append

        # Pop the first line of remaining_list and append it to collect_list
        #sed -i '1d' completed_list.cjr
        if [ -f collect_list.cjr ];then
        echo $COUNTER >> collect_list.cjr
        else
        echo "CJ::Reduce:: CJ in AWE. collect_list.cjr does not exists when CJ expects it." 1>&2
        exit 1
        fi

        percent_done=$(awk "BEGIN {printf \"%.2f\",100*${PROGRESS}/${TOTAL}}")
        printf "\n SubPackage %d Collected (%3.2f%%)" $COUNTER $percent_done

    done
    printf "\n"
    
fi
    
    
