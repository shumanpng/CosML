#!/usr/bin/env bash


TESTSET='cars'
DATADIR='../../../CrossDomainFewShot_original/filelists/'

NSHOT=5
NWAY=5

time python meta_train.py --testset $TESTSET \
                          --n_shot $NSHOT \
                          --train_n_way $NWAY \
                          --test_n_way $NWAY \
                          --task_network conv2+linear \
                          --feature_extractor_model conv4 \
                          --distance_metric euclidean \
                          --name cosml \
                          --save_dir ./output/ \
                          --data_dir $DATADIR \
                          --splits_dir ../data/crossdomain_data/ \
                          --save_freq 25 \
                          --train_aug \
                          --start_epoch 0 \
                          --stop_epoch 400 \
                          --mixed_task_batch_size 25 \
                          --pure_task_batch_size 25 \
                          --mixed_val_batch_size 5 \
                          --pure_val_batch_size 5 \
                          --pretrained_feature_extractor ./output/ptminiimagenet/399.tar

# to resume training from the last checkpont, add:  --resume true --resume_epoch -1