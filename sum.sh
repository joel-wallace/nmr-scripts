#!/bin/bash

SECOND=$((FIRST + 1))

tcsh <<EOF
    addNMR -in1 $WD/$FIRST/test.fid \
    -in2 $WD/$SECOND/test.fid   \
    -out $WD/test.fid
EOF

echo Added $FIRST and $SECOND

EXPS=$(seq $((SECOND + 1)) $LAST)

for i in $EXPS;
do
    tcsh <<EOF
        source ~/.tcshrc
        addNMR -in1 $WD/$i/test.fid \
        -in2 $WD/test.fid   \
        -out $WD/temp.fid
EOF
    mv "$WD/temp.fid" "$WD/test.fid"
    echo Added $i
done

echo "Final sum at $WD/test.fid"
