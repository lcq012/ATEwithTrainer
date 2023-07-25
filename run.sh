export CUDA_VISIBLE_DEVICES=4

for dataset in res14
do
    for seed in 456 789 345 2023 888
    do
        python3 ./main.py \
            --dataset_name ${dataset} \
            --seed ${seed}\
            --convall_file ./draft \
            --train_file ./raw_data/${dataset}train.json \
            --dev_file ./raw_data/${dataset}dev.json \
            --test_file ./raw_data/${dataset}test.json \
            --plm_path bert-base-uncased \
            --logging_dir ./logs\
            --max_length 110 \
            --plm_path ../PLM/bert-base-uncased \
            --train_batch_size 12 \
            --dev_batch_size 12 \
            --test_batch_size 12 \
            --learning_rate 3e-5 \
            --eval_steps 50 \
            --num_epochs 12 \
            --num_labels 3\
            --num_warmup_steps 100 \
            --task_name "ATE" 
    done
done