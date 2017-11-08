#!/usr/bin/env bash

NOW=$(date +%s)

for d in [0-9]*/;
do
	for f in $d/*.o*;
	do
		cat $f >> $d/results.csv
	done
	for f in $d/*.[o,e]*;
	do
		rm $f
	done
done

mkdir ./results/$NOW
mv [0-9]* ./results/$NOW/.
