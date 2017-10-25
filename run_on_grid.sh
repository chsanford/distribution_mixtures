#!/usr/bin/env bash

SIZE=7
POLICIES=$(((1<<$SIZE)-1))
ITER=`expr $SIZE - 1`

OUTFILE="$(date +%m-%d-%H%M%S).csv"
echo -n "train_policy, greedy_policy, err," >> $OUTFILE
QS=(q_L q_R)
for q in ${QS[@]};
do
    for i in `seq 0 $ITER`;
    do
        echo -n " $q$i," >> $OUTFILE
    done
done
WEIGHTS=(w_L w_R)
for w in ${WEIGHTS[@]};
do
    for j in `seq 0 3`;
    do
        echo -n " $w$j," >> $OUTFILE
    done
done
DS=(d_L d_R)
for d in ${DS[@]};
do
    for i in `seq 0 $ITER`;
    do
        echo -n " $d$i," >> $OUTFILE
    done
done


for t in `seq 1 $1`;
do
    for p in `seq 0 $POLICIES`;
    do
        qsub -l short -cwd ./python_exp.sh $p $OUTFILE $SIZE
    done
done