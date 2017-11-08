#!/usr/bin/env bash

source run_config.sh
source /gpfs/main/home/ctrimbac/envs/tensorflow/bin/activate

for r in `seq 1 $NUMEXPS`
do
    while IFS='' read -r line || [[ -n "$SEED" ]]; do
        mkdir $SEED
        python -c "import random_mdp_exp as r; r.exp_file($SIZE, $ACTIONS, $SEED)"
        cd $SEED
        OUTFILE="results.csv"
        echo -n "train_policy, greedy_policy, err" >> $OUTFILE
        QS=(q_L q_R)
        for q in ${QS[@]};
        do
            for i in `seq 0 $ITER`;
            do
                echo -n ", $q$i" >> $OUTFILE
            done
        done
        WEIGHTS=(w_L w_R)
        for w in ${WEIGHTS[@]};
        do
            for j in `seq 0 $POLYDIM`;
            do
                echo -n ", $w$j" >> $OUTFILE
            done
        done
        DS=(d_L d_R)
        for d in ${DS[@]};
        do
            for i in `seq 0 $ITER`;
            do
                echo -n ", $d$i" >> $OUTFILE
            done
        done

        echo "" >> $OUTFILE

        for t in `seq 1 $NUMTRIALS`;
        do
            for p in `seq 0 $POLICIES`;
            do
                qsub -l short -cwd ../python_exp.sh $SIZE $ACTIONS $SEED $p
                COUNT=$((COUNT+1))
            done
        done
        cd ..
    done < "rerun.txt"
done
echo "$COUNT jobs submitted."
