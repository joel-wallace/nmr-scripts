#!/usr/bin/bash

cd $WD

$PYTHON $MASTER/bootstrap.py $NUM_PEAKS < $BASELINED_NAME > $FIT_NAME


gnuplot -p <<EOF
    set title "$FID_NAME"
    set xlabel '19F (ppm)'
    set xrange [-60:-65]
    set grid ytics lc rgb "#eeeeee"
    
    set multiplot layout 2,1
    
    set size 1.0, 0.7
    set origin 0.0, 0.3
    set ylabel 'Intensity'
    set xlabel '19F (ppm)'
    
    plot "$BASELINED_NAME" u 1:2 w lines lc rgb "#bbbbbb" lw 1 title "Exp. Data", \
         "$FIT_NAME" u 1:2 w lines lc rgb "blue" lw 2 title "Total Fit", \
         "$FIT_NAME" u 1:4 w lines dt 2 lc rgb "red" title "Peak 1", \
         "$FIT_NAME" u 1:5 w lines dt 2 lc rgb "green" title "Peak 2", \
         "$FIT_NAME" u 1:6 w lines dt 2 lc rgb "orange" title "Peak 3", \
         "$FIT_NAME" u 1:7 w lines dt 2 lc rgb "purple" title "Peak 4"
    
    set size 1.0, 0.3
    set origin 0.0, 0.0
    set ylabel 'Residuals'
    unset xlabel
    plot "$FIT_NAME" u 1:3 w lines lc rgb "gray" title "Residuals"
    
    unset multiplot
EOF

