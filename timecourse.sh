#!/bin/bash

export STRIDE=4

EXPS=($(seq $FIRST $LAST))
echo "${#EXPS[@]} exps, $STRIDE at a time"

COUNTER=0 # the number of the experiment
TRACKER=0 # tracking the group of 3
NEWGROUP=false
rm -r $WD/timecourse
mkdir -p $WD/timecourse
while [ $COUNTER -lt ${#EXPS[@]} ]; do
    EXP=$(($COUNTER+$FIRST))
    # every 3 iterations, starting from the first one, this is true
    if [ $(($COUNTER % $STRIDE)) -eq 0 ]; then
        if [ $COUNTER -ne 0 ]; then echo "finished test$TRACKER.fid"; fi
        # the tracker increases every 3 exps
        let TRACKER=$(($TRACKER + $STRIDE))
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

cd $WD/timecourse/

bltxt_files=""

for fid in *.fid; do
    ft1=$(echo $fid | sed 's/\.fid$/.ft1/'); 
    txt=$(echo $fid | sed 's/\.fid$/.txt/'); 
    bltxt=$(echo $fid | sed 's/\.fid$/_bl.txt/'); 
    tcsh <<EOF
        nmrPipe -in $fid \
        | nmrPipe -fn EM -lb $LB -c 0.5               \
        | nmrPipe  -fn ZF -auto                        \
        | nmrPipe  -fn FT -auto                        \
        | nmrPipe  -fn PS -p0 $P0 -p1 $P1 -di -verb   \
        | nmrPipe -fn BASE -nw 10 -nl -70ppm -50ppm    \
        | nmrPipe -fn EXT -x1 -70ppm -xn -50ppm -sw    \
           -ov -out $ft1
        echo $ft1
        pipe2txt.tcl -index ppm $ft1 > $txt
        echo $txt
EOF
    $PYTHON $MASTER/baseline.py $EXCLUDE_LEFT $EXCLUDE_RIGHT < $txt > $bltxt
    echo $bltxt
    bltxt_files="$bltxt_files $bltxt"
done
echo $bltxt_files
gnuplot -p <<EOF
    set xrange [-60:-64]
    plot for [file in "$bltxt_files"] file with lines title file
EOF
