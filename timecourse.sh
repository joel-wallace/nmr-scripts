#!/bin/bash

export STRIDE=3

EXPS=($(seq $FIRST $LAST))
echo "${#EXPS[@]} exps, $STRIDE at a time"

COUNTER=0 # the number of the experiment
TRACKER=0 # tracking the group of 3
NEWGROUP=false
mkdir -p $WD/timecourse
while [ $COUNTER -lt ${#EXPS[@]} ]; do
    EXP=$(($COUNTER+$FIRST))
    # every 3 iterations, starting from the first one, this is true
    if [ $(($COUNTER % $STRIDE)) -eq 0 ]; then
        if [ $COUNTER -ne 0 ]; then echo "finished test$TRACKER.fid"; fi
        # the tracker increases every 3 exps
        let TRACKER=TRACKER+3
        cp $WD/$EXP/test.fid $WD/timecourse/test$TRACKER.fid
        echo "started test$TRACKER.fid"
    else
        tcsh <<EOF
            source ~/.tcshrc
            addNMR -in1 $WD/timecourse/test$TRACKER.fid \
                -in2 $WD/$EXP/test.fid \
                -out $WD/timecourse/temp.fid
            mv $WD/timecourse/temp.fid $WD/timecourse/test$TRACKER.fid
EOF
    fi

    # move to the next exp
    let COUNTER=COUNTER+1
done
exit
