#!/bin/tcsh

cd $WD/$FIRST
mv ./fid.com ./fid.com.bak
bruker
sed -i '/nmrPipe/d' ./fid.com
sed '/sleep/d' ./fid.com > $MASTER/bruk2pipe.com
