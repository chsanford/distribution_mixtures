#!/usr/bin/env bash

RERUNFILE="rerun.txt"

for d in *[0-9]/;
do
    SUB="$(grep -c ', suboptimal' $d/*.o*)"
    OPT="$(grep -c ', optimal' $d/*.o*)"
    if (( OPT > SUB ));
    then
        echo $d >> $RERUNFILE
    fi
done