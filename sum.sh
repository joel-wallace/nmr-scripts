#!/bin/bash

mkdir -p $OUT_DIR

SECOND=$((FIRST + 1))

tcsh <<EOF
    addNMR -in1 $WD/$FIRST/test.fid \
    -in2 $WD/$SECOND/test.fid   \
    -out $OUT_DIR/test.fid
EOF

echo Added $FIRST and $SECOND

EXPS=$(seq $((SECOND + 1)) $LAST)

for i in $EXPS;
do
    tcsh <<EOF
        source ~/.tcshrc
        addNMR -in1 $WD/$i/test.fid \
        -in2 $OUT_DIR/test.fid   \
        -out $OUT_DIR/temp.fid
EOF
    mv "$OUT_DIR/temp.fid" "$OUT_DIR/test.fid"
    echo Added $i
done

echo "Final sum at $OUT_DIR/test.fid"
