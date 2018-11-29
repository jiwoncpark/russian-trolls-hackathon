#!/bin/bash -l


if [ ! -f "/home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/collect_list.cjr" ];then
#build a file of jobs
seq 25 > /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/job_list.cjr
cp   /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/job_list.cjr  /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/remaining_list.cjr
else
grep -Fxvf /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/collect_list.cjr /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/job_list.cjr  >  /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/remaining_list.cjr;
fi

    
if [ -f "/home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/completed_list.cjr" ];then
    rm /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/completed_list.cjr
fi
    
    
touch completed_list.cjr
for line in $(cat /home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/remaining_list.cjr);do
COUNTER=`grep -o "[0-9]*" <<< $line`
if [ -f "/home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/$COUNTER/results/training_results.csv" ];then
echo -e "$COUNTER\t" >> "/home/zyflame104/CJRepo_Remote/train/5a57aee690751a78ed57169f146e4977d1042a75/completed_list.cjr"
fi
done
    
    
