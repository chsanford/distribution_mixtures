#!/usr/bin/env bash

RERUNFILE="rerun.txt"
rm -f $RERUNFILE
for d in [0-9]*/;
do
    SUB="$(cat $d*.o* | grep -c ', suboptimal')"
    OPT="$(cat $d*.o* | grep -c ', optimal')"
    if (( OPT > SUB ));
    then
        echo "${d%/}" >> $RERUNFILE
    fi
done

source move_results.sh