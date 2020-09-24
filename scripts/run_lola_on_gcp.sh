NPROC=$(nproc)
echo "$NPROC"
for j in $(seq 0 $1); do
    START=$(($j*$NPROC+1))
    END=$(($j*$NPROC+$NPROC))
    for i in $(seq $START $END); do 
        echo "Starting Job $i"
        python3 run_lola.py --exp_name=CoinGame --no-exact --trace_length=20 --run_id=$i --num_episodes=$2 --batch_size=$3 & 
        done
    wait
    done

wait
echo "Finished"