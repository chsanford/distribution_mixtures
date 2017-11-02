#!/usr/bin/env bash

SIZE=7
ACTIONS=2
POLICIES=$(((1<<$SIZE)-1))
ITER=`expr $SIZE - 1`
NUMTRIALS=$2
NUMEXPS=$1
COUNT=0

for r in `seq 1qstat $NUMEXPS`
do
    SEED=$RANDOM
    qstatpython -c 'import random_mdp_exp as r; r.exp_file($SIZE, $ACTIONS, $SEED)'
    mkdir $SEED
    cd $SEED
    OUTFILE="results.csv"
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


    for t in `seq 1 $NUMTRIALS`;
    do
#        for p in `seq 0 $POLICIES`;
#        do
#            qsub -l short -cwd ../python_exp.sh $SIZE $ACTIONS $SEED $p
#        done
        qsub -l short -cwd ../python_exp.sh $SIZE $ACTIONS $SEED
        COUNT=$((COUNT+1))
    done
    cd ..
done
echo "$COUNT jobs submitted."
