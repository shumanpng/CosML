# This code is modified from https://github.com/wyharveychen/CloserLookFewShot/blob/master/test.py

import numpy as np
import torch

from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import datetime
import glob
import copy

import configs
import maml.models.conv_net as convnet
import maml.models.backbone as backbone

from data.datamgr import SetDataManager

from methods.maml import MAML
from options import parse_args, get_best_file, get_assigned_file, get_assigned_file_exact_path

from data.domain import Domain


def evaluate(test_loader, model, seen_domains):
    acc, loss, acc_all = model.mixed_test_loop(test_loader, seen_domains, verbose = False, test = True)
    return acc_all



if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('test')

    params.model = configs.options['meta_test']['model']
    params.method = configs.options['meta_test']['method']

    if params.dataset == 'multi':
        params.dataset = ['miniImagenet', 'cars', 'cub', 'places', 'plantae']
    else:
        params.dataset = params.dataset.split(',')
    print("params.dataset: {}\n".format(params.dataset))

    params.n_episode = 10
    params.gradient_steps = 15
    datasets = params.dataset
    if params.testset in params.dataset:
        datasets.remove(params.testset)

    params.init_num_channels = 64
    params.fix_num_channels = True

    params.name = params.name + '_' + '_'.join(datasets) + '_{}w{}s_{}'.format(params.train_n_way, params.n_shot, params.feature_extractor_model)
    params.name = params.name + '_{}_{}'.format(params.task_network, params.distance_metric)

    print("Testing! {}-way {}-shot on {} dataset with {} epochs of CosML model ({}-{})".format( params.test_n_way, params.n_shot,
                                                                                            params.testset,
                                                                                            params.save_epoch,
                                                                                            params.feature_extractor_model,
                                                                                            params.task_network))

    print("\nTesting on {} dataset".format(params.testset))
    print("Model trained on {} datasets".format(datasets))
    print("# of gradient update steps per task: {}".format(params.gradient_steps))
    print("init_num_channels: {}".format(params.init_num_channels))
    print("fix_num_channels: {}".format(params.fix_num_channels))
    print("feature extractor model: {}\n".format(params.feature_extractor_model))

    ## Evaluate ##
    acc_all = []
    iter_num = 100

    ## Load test data ##
    splits_dir = params.splits_dir
    image_size = 84
    n_query = 15 ## setting used in CrossDomainFewShot and CloserLookFewShot
    test_file = os.path.join(splits_dir, params.testset, 'novel.json') # can test on either base, val, or novel since this is an unseen dataset
    test_datamgr = SetDataManager(image_size, n_way = params.test_n_way, n_support = params.n_shot,
                                    n_query = n_query, n_episode = params.n_episode)
    test_loader = test_datamgr.get_data_loader(params.data_dir, test_file, aug = False)

    ## Load pre-trained feature extractor (to be held fixed) ##
    convnet.ConvBlock.maml = True
    if params.feature_extractor_model == 'conv4':
        conv_net = convnet.ConvNet(depth = 4, init_num_channels = params.init_num_channels,
                                    fix_num_channels = params.fix_num_channels,
                                    flatten = True)
        if params.init_num_channels == 64 and params.fix_num_channels:
            conv_net.final_feat_dim = 1600
        elif params.init_num_channels == 32 and not params.fix_num_channels:
            ## for variable channels ##
            conv_net.final_feat_dim = 6400 ## obtained by outputting the shape of the features from the final conv layer

        pretrained_model = convnet.PreTrain(model_func = conv_net,
                                    num_class = params.num_classes,
                                    tf_path = None)

    elif params.feature_extractor_model == 'resnet10':
        resnet = backbone.ResNet10()
        pretrained_model = convnet.PreTrain(model_func = resnet, num_class = params.num_classes,
                                            tf_path = None)

    modelfile = get_assigned_file_exact_path(params.pretrained_feature_extractor)
    print("pretrained feature extractor model file: {}".format(modelfile))
    if modelfile is not None:
        tmp = torch.load(modelfile)
        try:
            pretrained_model.load_state_dict(tmp['state'])
        except RuntimeError:
            print("warning! RuntimeError when load_state_dict()")
            pretrained_model.load_state_dict(tmp['state'], strict=False)
        except KeyError:
            for k in tmp['model_state']:
                if 'running' in k:
                    tmp['model_state'][k] = tmp['model_state'][k].squeeze()
            pretrained_model.load_state_dict(tmp['model_state'], strict=False)
        except:
            raise
    print("succesfully loaded pre-trained model (feature extractor)\n")

    if params.feature_extractor_model == 'conv4':
        if params.task_network == 'conv1+linear':
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-2]
                break
        elif params.task_network == 'conv2+linear':
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-3]
    elif params.feature_extractor_model == 'resnet10':
        if params.task_network == 'conv1+linear':
            # 4th last layer: layer 6 (simple block) - inchannel 128, out_channel 256
            # remove last 3 layers for now
            # 3rd last layer: layer 7 (simpleblock) - in channel 256, out_channel 512
            # 2nd last layer: layer 8 (avg pooling)
            # last layer: layer 9 (flatten)
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-3]
                break
        elif params.task_network == 'linear':
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-1] # only remove flatten layer
                break
        else:
            print("ERROR: {} task network not implemented.".format(params.task_network))
    feature_extractor_g = torch.nn.Sequential(*intermediate_feature_layers)

    print("feature extractor g:")
    if params.feature_extractor_model == 'conv4':
        for blocknum, convblock in enumerate(list(feature_extractor_g.children())):
            for param in convblock.C.parameters():
                param.requires_grad = False
            for param in convblock.BN.parameters():
                param.requires_grad = False
            for param in convblock.relu.parameters():
                param.requires_grad = False
            for param in convblock.pool.parameters():
                param.requires_grad = False

        print("\n\n--- setting feature_extractor (ConvNet) as pre-trained feature extractor---\n\n")
        if params.task_network == 'conv1+linear':
            feature_extractor = convnet.ConvNet(depth = 3, init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels, flatten = False)
            feature_extractor.trunk = feature_extractor_g
        elif params.task_network == 'conv2+linear':
            feature_extractor = convnet.ConvNet(depth = 2, init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels, flatten = False)
            feature_extractor.trunk = feature_extractor_g
        elif params.task_network == 'linear':
            feature_extractor = convnet.ConvNet(depth = 4, init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels, flatten = False)
            feature_extractor.trunk = feature_extractor_g

    elif params.feature_extractor_model == 'resnet10':
        for blocknum, convblock in enumerate(list(feature_extractor_g.children())):
            print("\n-- ConvBlock: {} --".format(blocknum))
            # print(convblock)
            for param in convblock.named_parameters():
                param[1].requires_grad = False
                print("param {} | requires_grad: {}".format(param[0], param[1].requires_grad))

        feature_extractor = backbone.ResNet10() ## CHECK THIS LATER!!
        feature_extractor.trunk = feature_extractor_g

    ## init domain specific models ##
    seen_domains = [Domain(dataset=dataset) for dataset in datasets]

    if params.feature_extractor_model == 'conv4':
        conv1_channels = (128, 256)
        if params.fix_num_channels and params.init_num_channels == 64:
            conv1_channels = (64, 64)
    elif params.feature_extractor_model == 'resnet10':
        conv1_channels = (256, 256)

    for domain in seen_domains:
        convnet.ConvBlock.maml = True
        if params.task_network == 'linear':
            maml_convnet = None
        elif params.task_network == 'conv1+linear' or params.task_network == 'conv2+linear':
            if params.task_network == 'conv1+linear':
                maml_convnet = convnet.ConvNet(depth=1, preset_in_out_channels = conv1_channels,
                                                init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels,
                                                flatten = True)
            elif params.task_network == 'conv2+linear':
                maml_convnet = convnet.ConvNet(depth=2, preset_in_out_channels = conv1_channels,
                                                init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels,
                                                flatten = True)

        if params.feature_extractor_model == 'resnet10':
            if params.task_network == 'conv1+linear':
                if conv1_channels == (256, 256):
                    maml_convnet.final_feat_dim = 2304
                elif conv1_channels == (256, 512):
                    maml_convnet.final_feat_dim = 4608
        if params.feature_extractor_model == 'conv4':
            if params.init_num_channels == 64 and params.fix_num_channels:
                maml_convnet.final_feat_dim = 1600
            elif params.init_num_channels == 32 and not params.fix_num_channels:
                ## for variable channels ##
                maml_convnet.final_feat_dim = 6400

        domain.model = MAML(pretrained_feature_extractor = feature_extractor,
                            model_info = (params.feature_extractor_model, params.task_network),
                            distance = params.distance_metric,
                            model_func = maml_convnet, n_way = params.train_n_way,
                            n_support = params.n_shot)
        domain.model = domain.model.cuda()

    ## init model ##
    if params.task_network == 'conv1+linear' or params.task_network == 'conv2+linear':
        if params.task_network == 'conv1+linear':
            print("using fixed pretrained feature extractor (conv3) for maml's task network (conv1 + linear layer)")
            maml_convnet = convnet.ConvNet(depth = 1, preset_in_out_channels = conv1_channels,
                                            init_num_channels = params.init_num_channels,
                                            fix_num_channels = params.fix_num_channels,
                                            flatten = True)
        elif params.task_network == 'conv2+linear':
            print("using fixed pretrained feature extractor (conv2) for maml's task network (conv2 + linear layer) ")
            maml_convnet = convnet.ConvNet(depth=2, preset_in_out_channels = conv1_channels,
                                            init_num_channels = params.init_num_channels,
                                            fix_num_channels = params.fix_num_channels,
                                            flatten = True)

    if params.feature_extractor_model == 'resnet10':
        if params.task_network == 'conv1+linear':
            if conv1_channels == (256, 256):
                maml_convnet.final_feat_dim = 2304
            elif conv1_channels == (256, 512):
                maml_convnet.final_feat_dim = 4608
    if params.feature_extractor_model == 'conv4':
        if params.init_num_channels == 64 and params.fix_num_channels:
            maml_convnet.final_feat_dim = 1600
        elif params.init_num_channels == 32 and not params.fix_num_channels:
            ## for variable channels ##
            maml_convnet.final_feat_dim = 6400

    model = MAML(pretrained_feature_extractor = feature_extractor,
                        model_info = (params.feature_extractor_model, params.task_network),
                        distance = params.distance_metric,
                        model_func = maml_convnet, n_way = params.train_n_way,
                        n_support = params.n_shot)
    model.task_update_num = params.gradient_steps
    model = model.cuda()



    ## Load model + saved prototypes ##
    model_name = 'cosml'
    if params.feature_extractor_model == 'resnet10':
        model_name = 'cosml_resnet10'

    elif params.feature_extractor_model == 'conv4':
        if params.fix_num_channels:
            model_name = 'cosml_fixed_channels{}'.format(params.init_num_channels)

    if params.meta_learner_path != '':
        checkpoint_dir = params.meta_learner_path
    else:
        checkpoint_dir = '{}/{}/checkpoints/{}'.format(params.save_dir, model_name, params.name)
    print("loading trained metalearners from: {}\n".format(checkpoint_dir))
    for domain in seen_domains:
        dataset = domain.dataset

        if params.save_epoch != -1:
            modelfile = get_assigned_file(checkpoint_dir = os.path.join(checkpoint_dir, dataset), num = params.save_epoch)
        else: # load best model
            modelfile = get_best_file(os.path.join(checkpoint_dir, dataset))
        if modelfile is not None:
            tmp = torch.load(modelfile)
            try:
                domain.model.load_state_dict(tmp['state'])
            except RuntimeError:
                print('warning! RuntimeError when load_state_dict{}!')
                domain.model.load_state_dict(tmp['state'], strict = False)
            except KeyError:
                for k in tmp['model_state']:
                    if 'running' in k:
                        tmp['model_state'][k] = tmp['model_state'][k].squeeze()
                domain.model.load_state_dict(tmp['model_state'], strict = False)
            except:
                raise
        domain.prototypes = tmp['prototypes']
        print("{} loaded domain prototype: {}".format(dataset, domain.prototypes['domain']))
        print("{} loaded task prototypes count: {}".format(dataset, len(domain.prototypes['task'])))
        domain.model.eval()


    print("\nnumber of testing iterations (tasks): {}".format(iter_num))
    for i in range(iter_num):
        print_freq = 1
        acc = evaluate(test_loader, model, seen_domains)
        acc_all.extend(acc)
        if (i % print_freq) == 0 or i == iter_num - 1:
            print("iter {}/{} - acc: {}".format(i, iter_num - 1, np.mean(acc)))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    total_tasks = iter_num * params.n_episode
    print("{} test iterations - {} tasks per iter: Acc = {:4.2f}% +- {:4.2f}%".format(iter_num, params.n_episode, acc_mean, 1.96*acc_std/np.sqrt(total_tasks)))

    model_name = 'cosml_fixed_channels64'
    test_results_dir = '{}/{}/test_results/{}'.format(params.save_dir, model_name, params.name)
    if not os.path.isdir(test_results_dir):
        os.makedirs(test_results_dir)
    results_file = os.path.join(test_results_dir, '{}_testset_{}_results.txt'.format(params.name, params.testset))

    with open(results_file, 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = 'training datasets: {} | test dataset: {} | {}-{} {}shot {}way_train {}way_test'.format(datasets, params.testset, params.model, params.method,
                                                                                                                params.n_shot, params.train_n_way, params.test_n_way)
        acc_str = "{} test iterations - {} tasks per iter: Acc = {:4.2f}% += {:4.2f}%".format(iter_num, params.n_episode, acc_mean, 1.96*acc_std/np.sqrt(total_tasks))
        f.write('Time: {}\nSetting: {}\nAcc: {}\n\n'.format(timestamp, exp_setting, acc_str))
