#!/usr/bin/env bash

TESTSET='places'
DATADIR='../../../CrossDomainFewShot_original/filelists/'
MODELNAME='metalearners_ptminiimagenet399_gconvmaml_miniImagenet_cars_cub_plantae_5w5s_conv4_conv2+linear_euclidean'

time python meta_test.py --testset $TESTSET \
                        --feature_extractor_model conv4 \
                        --task_network conv2+linear \
                        --distance_metric euclidean \
                        --name metalearners_ptminiimagenet399_gconvmaml \
                        --save_dir output/ \
                        --data_dir $DATADIR \
                        --splits_dir ../data/crossdomain_data/ \
                        --train_n_way 5 \
                        --test_n_way 5 \
                        --n_shot 5 \
                        --meta_learner_path ./output/$MODELNAME \
                        --save_epoch 319 \
                        --pretrained_feature_extractor ./output/ptminiimagenet/399.tar \
