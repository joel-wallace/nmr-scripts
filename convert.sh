#!/bin/bash

SCRIPT=$CONVERSION_SCRIPT

EXPS=$(seq $FIRST $LAST)

for i in $EXPS;
do
    printf "\n\nProcessing $i:"
    cd $WD/$i
    [ -f $SCRIPT ] && mv ./$SCRIPT ./$SCRIPT.bak
    cp $MASTER/$SCRIPT .
    chmod +x ./$SCRIPT
    ./$SCRIPT
done
