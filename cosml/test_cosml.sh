#!/usr/bin/env bash

TESTSET='cars'
DATADIR='../../../CrossDomainFewShot_original/filelists/'
MODELNAME='cosml_miniImagenet_cub_places_plantae_5w5s_conv4_conv2+linear_euclidean'

NSHOT=5
NWAY=5

time python meta_test.py --testset $TESTSET \
                        --feature_extractor_model conv4 \
                        --task_network conv2+linear \
                        --distance_metric euclidean \
                        --name metalearners_ptminiimagenet399_gconvmaml \
                        --save_dir output \
                        --data_dir $DATADIR \
                        --splits_dir ../data/crossdomain_data/ \
                        --train_n_way $NWAY \
                        --test_n_way $NWAY \
                        --n_shot $NSHOT \
                        --meta_learner_path ./output/checkpoints/$MODELNAME \
                        --save_epoch 319 \
                        --pretrained_feature_extractor ./output/ptminiimagenet/399.tar \
