#!/bin/bash
make -j8
exe="thinexec"
imgdir="../../img"
imgs=`ls $imgdir`

for img in $imgs; do
    ./$exe $imgdir/$img
done
