#!/bin/tcsh

cd $WD/$FIRST
mv ./fid.com ./fid.com.bak
bruker
sed '/sleep/d' ./fid.com > $MASTER/bruk2pipe.com
