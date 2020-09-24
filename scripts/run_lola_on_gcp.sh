NPROC=$(nproc)
echo "$NPROC"
LESS=0
END=$(($NPROC-$LESS))

for i in $(seq 1 $END); do 
    echo "Starting Job $i"
    python3 run_lola.py --exp_name=CoinGame --no-exact --trace_length=20 --run_id=$i --num_episodes=$1 --batch_size=$2 & 
    done

wait
echo "Finished"