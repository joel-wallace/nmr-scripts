#!/bin/bash

SCRIPT=$PROC_SCRIPT

echo "Processing $WD/test.fid"

cd $WD

NORM=$(($NS * $SAMPLE_CONC))

# LB: Hz
# Zero-fill, double-size, round to power of 2
# Fourier transform
# Phase and delete imaginaries
# Baseline
# Extract range

tcsh <<EOF
    nmrPipe -in $FID_NAME \
    | nmrPipe -fn EM -lb $LB -c 0.5              \
    | nmrPipe -fn MULT -c $NORM -inv             \
    | nmrPipe -fn ZF -auto                       \
    | nmrPipe -fn FT -auto                       \
    | nmrPipe -fn PS -p0 $P0 -p1 $P1 -di -verb   \
    | nmrPipe -fn BASE -nw 10 -nl -70ppm -50ppm  \
    | nmrPipe -fn EXT -x1 -70ppm -xn -50ppm -sw  \
       -ov -out $FT1_NAME
    
    pipe2txt.tcl -index ppm $FT1_NAME > $OUT_NAME
EOF

echo "Written to $WD/$OUT_NAME"

#gnuplot -p <<EOF
#set title "$OUT_NAME"
#set xlabel 'ppm'
#set xrange [-50:-70]
#plot "$OUT_NAME" with lines title 'Processed Spectrum'
#EOF

$PYTHON $MASTER/baseline.py $EXCLUDE_LEFT $EXCLUDE_RIGHT < $OUT_NAME > $BASELINED_NAME

echo "Written to $WD/$BASELINED_NAME"

gnuplot -p <<EOF
    set title "$BASELINED_NAME"
    set xlabel 'ppm'
    set xrange [-55:-70]
    plot "$BASELINED_NAME" with lines lc rgb "gray" title "Baseline", \
         "$BASELINED_NAME" using 1:((\$1 <= $EXCLUDE_LEFT && \$1 >= $EXCLUDE_RIGHT) ? \$2 : 1/0) \
         with lines lc rgb "red" lw 2 title "Peak Region"
EOF
