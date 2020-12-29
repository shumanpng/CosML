# This code is modified from https://github.com/wyharveychen/CloserLookFewShot/blob/master/train.py
import numpy as np
import torch

from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from datetime import datetime
import os
import glob
import copy

import configs
# import backbone
import maml.models.conv_net as convnet
import maml.models.backbone as backbone


from data.datamgr import SetDataManager

from methods.maml import MAML
from options import parse_args, get_resume_file, get_assigned_file, get_assigned_file_exact_path

from data.domain import Domain


def update_domain_prototype(original_proto, new_proto):
    """
    original_proto:
        {'count': number of examples (images) used in the computation of the
                  current domain prototype (mean),
         'mean': value of the current feature mean over 'count' number of examples }

    new_proto:
        {'count': number of examples used in the computation of this new prototype,
         'mean': value of the current feature mean over 'count' number of examples}

    return:
    updated_proto:
        {'count': original_proto['count'] + new_proto['count'],
         'mean': mean(original_proto['count']*original_proto['mean'],
                        new_proto['count']*new_proto['mean'])
    """
    updated_proto = {}
    cp_original_proto_count = original_proto['count']
    cp_original_proto_mean = original_proto['mean'].clone() if original_proto['mean'] is not None else None
    cp_new_proto_count = new_proto['count']
    cp_new_proto_mean = new_proto['mean'].clone()

    if cp_original_proto_mean is None and cp_original_proto_count == 0:
        updated_proto['count'] = cp_new_proto_count
        updated_proto['mean'] = cp_new_proto_mean
    else:
        total_count = cp_original_proto_count + cp_new_proto_count
        original_total = float(cp_original_proto_count) * cp_original_proto_mean
        new_total = float(cp_new_proto_count) * cp_new_proto_mean
        updated_proto['count'] = total_count
        updated_proto['mean'] = (1/float(total_count)) * (original_total + new_total)

    return updated_proto

def train(log_file, seen_domains, mixed_base_loader, mixed_val_loader, model, start_epoch, stop_epoch, params):

    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()))

    ## Add optimizer to Domain class ##
    for domain in seen_domains:
        domain.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, domain.model.parameters()))

    total_it = 0

    for epoch in range(start_epoch, stop_epoch):
        epoch_start_time = time.time()
        print("Training epoch {}/{}".format(epoch, stop_epoch-1))

        with open(log_file, 'a') as f:
            f.write("Epoch {:d} \n".format(epoch))


        ## PURE TASKS ##
        ## train batches of pure tasks from each domain ##
        for d_idx, domain in enumerate(seen_domains):
            domain_start_time = time.time()
            print("domain specific meta-training {}/{} | current domain: {}".format(d_idx, len(seen_domains)-1, domain.dataset))
            domain.model.train()
            domain.total_it, new_prototypes = domain.model.train_loop(epoch = epoch,
                                                                        train_loader = domain.train_dataloader,
                                                                        optimizer = domain.optimizer,
                                                                        total_it = domain.total_it,
                                                                        dataset = domain.dataset,
                                                                        compute_prototypes = True)
            domain.model.eval()
            original_proto = domain.prototypes['domain']
            new_proto = new_prototypes['domain']
            domain.prototypes['domain'] = update_domain_prototype(original_proto = original_proto,
                                                                                new_proto = new_proto)
            domain.prototypes['task'].extend(new_prototypes['task'])

            domain_end_time = time.time()
            domain_elapsed_time = domain_end_time - domain_start_time
            print("domain {} took {:2} min {:.2f} sec to train".format(d_idx, int(domain_elapsed_time//60), domain_elapsed_time%60))

            acc, loss, _ = domain.model.test_loop(domain.val_dataloader)
            print("{} - Epoch {} | Val Acc: {} | Val Loss: {}".format(domain.dataset, epoch, acc, loss))

            with open(log_file, 'a') as f:
                f.write("{}     Val Acc: {:f} | Val Loss: {:f}\n".format(domain.dataset, acc, loss))

            if acc > domain.max_acc:
                print("best {} model! save...".format(domain.dataset))
                domain.max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, domain.dataset, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict(), 'prototypes': domain.prototypes}, outfile)

            else:
                print("best accuracy so far: {:f}".format(domain.max_acc))

            if ((epoch + 1) % params.save_freq == 0 ) or (epoch == stop_epoch - 1):
                outfile = os.path.join(params.checkpoint_dir, domain.dataset, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict(), 'prototypes': domain.prototypes}, outfile)

        if params.mixed_task_training == 'yes':
            ## MIXED TASKS ##
            ## train batches of mixed tasks ##
            mixed_start_time = time.time()
            model.train()
            print("starting mixed task training ")
            total_it = model.mixed_train_loop(epoch = epoch, train_loader = mixed_base_loader, optimizer = optimizer,
                                                total_it = total_it, seen_domains = seen_domains)

            model.eval()
            mixed_end_time = time.time()
            mixed_elapsed_time = mixed_end_time - mixed_start_time
            print("mixed tasks took {:2} min {:.2}f sec to train".format(int(mixed_elapsed_time//60), mixed_elapsed_time%60))

            mixed_acc, mixed_loss, _ = model.mixed_test_loop(mixed_val_loader, seen_domains)

            with open(log_file, 'a') as f:
                f.write("Mixed task     Val Acc: {:f} | Val Loss: {:f}\n".format(mixed_acc, mixed_loss))

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        print("epoch {} took {:2} min {:.2f} sec in total\n".format(epoch, int(epoch_elapsed_time//60), epoch_elapsed_time%60))

    return model



if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    ## update params ##
    params.dataset = configs.options['meta_train']['dataset']
    params.model = configs.options['meta_train']['model']
    params.method = configs.options['meta_train']['method']
    params.train_type = configs.options['meta_train']['train_type']

    if params.pretrained_feature_extractor == '':
        raise Exception("ERROR: pretrained_feature_extractor is not specified!\n\n")

    params.init_num_channels = 64
    params.fix_num_channels = True


    print("\ninit_num_channels: {} | fix_num_channels: {} | use mixed tasks: {} | gconvmaml: {} | train_aug: {}\n".format(params.init_num_channels, params.fix_num_channels,
                                                                                            params.mixed_task_training, params.gconvmaml, params.train_aug) )


    ## STEP 1: init domains ##
    datasets = params.dataset
    testset_ls = params.testset.split(',')
    for testset in testset_ls:
        datasets.remove(testset)
    print("seen domains: {}".format(datasets))
    print("testset: {}\n".format(params.testset))
    seen_domains = [Domain(dataset=dataset) for dataset in datasets]
    for domain in seen_domains:
        print("Domain: {}".format(domain.dataset))
        print("train dataloader: {}".format(domain.train_dataloader))
        print("val dataloader: {}".format(domain.val_dataloader))
        print("prototypes: {}".format(domain.prototypes))
        print("model: {}\n".format(domain.model))



    ## output and tensorboard dir ##
    params.name = params.name + '_' + '_'.join(datasets) + '_{}w{}s_{}'.format(params.train_n_way, params.n_shot, params.feature_extractor_model)
    params.name = params.name + '_{}_{}'.format(params.task_network, params.distance_metric)
    model_name = 'cosml'
    if params.feature_extractor_model == 'resnet10':
        model_name = 'cosml_resnet10'

    elif params.feature_extractor_model == 'conv4':
        if params.fix_num_channels:
            model_name = 'cosml_fixed_channels{}'.format(params.init_num_channels)
    params.tf_dir = '{}/log/{}'.format(params.save_dir, params.name)
    params.checkpoint_dir = '{}/checkpoints/{}'.format(params.save_dir, params.name)
    for dataset in datasets:
        checkpoint_dir = os.path.join(params.checkpoint_dir, dataset)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    if not os.path.isdir(params.tf_dir):
        os.makedirs(params.tf_dir)

    print("params.name (model name): {}".format(params.name))


    if params.gconvmaml == 'yes':
        convnet.ConvBlock.maml = True

    ## Load pre-trained feature extractor (to be held fixed) ##
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
                intermediate_feature_layers = layer[:-3] ## double check this
        elif params.task_network == 'conv3+linear':
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-4]
                break
        elif params.task_network == 'linear':
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-1] # only remove classification layer
        else:
            print("ERROR: {} task network not implemented.".format(params.task_network))
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
            # keep the entire resnet10 as is
            for layer in pretrained_model.feature.children():
                intermediate_feature_layers = layer[:-1] # only remove flatten layer
                # intermediate_feature_layers = layer
                break
        else:
            print("ERROR: {} task network not implemented.".format(params.task_network))
    feature_extractor_g = torch.nn.Sequential(*intermediate_feature_layers)

    print("feature extractor g:")
    if params.feature_extractor_model == 'conv4':
        for blocknum, convblock in enumerate(list(feature_extractor_g.children())):
            ## Fix the weights in each layer of the conv block ##
            for param in convblock.C.parameters():
                param.requires_grad = False
            for param in convblock.BN.parameters():
                param.requires_grad = False
            for param in convblock.relu.parameters():
                param.requires_grad = False
            for param in convblock.pool.parameters():
                param.requires_grad = False

        convnet.ConvBlock.maml = True ## this is new; test conv4 with this new setting later

        print("\n\n--- setting feature_extractor (ConvNet) as pre-trained feature extractor---\n\n")
        if params.task_network == 'conv1+linear':
            feature_extractor = convnet.ConvNet(depth = 3, init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels, flatten = False)
            feature_extractor.trunk = feature_extractor_g
        elif params.task_network == 'conv2+linear':
            feature_extractor = convnet.ConvNet(depth = 2, init_num_channels = params.init_num_channels,
                                                fix_num_channels = params.fix_num_channels, flatten = False)
            feature_extractor.trunk = feature_extractor_g
        elif params.task_network == 'conv3+linear':
            feature_extractor = convnet.ConvNet(depth = 1, init_num_channels = params.init_num_channels,
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


    ## STEP 4 ##
    if params.feature_extractor_model == 'conv4':
        conv1_channels = (128, 256)
        if params.fix_num_channels and params.init_num_channels == 64:
            conv1_channels = (64, 64)
    elif params.feature_extractor_model == 'resnet10':
        conv1_channels = (256, 256) ## after removing last simple block layer in resnet10 with channels (256, 512)

    image_size = 84
    if params.feature_extractor_model == 'resnet10' and params.task_network == 'linear':
        image_size = 224 ## otherwise the image is too small for the complete resnet model
    params.n_query = max(1, int(16*params.test_n_way/params.train_n_way))

    for i, dataset in enumerate(datasets):
        base_file = os.path.join(params.splits_dir, dataset, 'base.json')
        val_file = os.path.join(params.splits_dir, dataset, 'val.json')

        ## create dataloader for each domain (pure tasks) ##
        train_few_shot_params = dict(n_way = params.train_n_way,
                                     n_support = params.n_shot)
        train_pt_datamgr = SetDataManager(image_size,
                                          n_query = params.n_query,
                                          mixed_tasks = False,
                                          n_episode = params.pure_task_batch_size,
                                          **train_few_shot_params)
        train_pt_loader = train_pt_datamgr.get_data_loader(params.data_dir,
                                                            base_file,
                                                            aug = params.train_aug )
        test_few_shot_params = dict(n_way = params.test_n_way,
                                    n_support = params.n_shot)
        val_pt_datamgr = SetDataManager(image_size,
                                        n_query = params.n_query,
                                        mixed_tasks = False,
                                        n_episode = params.pure_val_batch_size,
                                        **test_few_shot_params)
        val_pt_loader = val_pt_datamgr.get_data_loader(params.data_dir,
                                                        val_file,
                                                        aug = False)
        seen_domains[i].train_dataloader = train_pt_loader
        seen_domains[i].val_dataloader = val_pt_loader

        print("-- dataset: {} --".format(dataset))
        print("\n\ninitializing maml model for task network... ")
        convnet.ConvBlock.maml = True ## this is new; test conv4 with this new setting later
        if params.task_network == 'linear':
            maml_convnet = None
        elif params.task_network == 'conv1+linear' or params.task_network == 'conv2+linear' or params.task_network == 'conv3+linear':
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
            elif params.task_network == 'conv3+linear':
                maml_convnet = convnet.ConvNet(depth = 3, preset_in_out_channels = conv1_channels,
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
                    maml_convnet.final_feat_dim = 6400 ## obtained by outputting the shape of the features from the final conv layer


        seen_domains[i].model = MAML(pretrained_feature_extractor = feature_extractor,
                                     model_info = (params.feature_extractor_model, params.task_network),
                                     model_func = maml_convnet, **train_few_shot_params,
                                     distance = params.distance_metric,
                                     method = 'CosML', tf_path = params.tf_dir)
        seen_domains[i].model = seen_domains[i].model.cuda()
        print("succesfully initialized maml model for task network!\n")


    ## Mixed tasks dataloader ##
    print("-- creating mixed tasks dataloaders --")

    base_file = [os.path.join(params.splits_dir, dataset, 'base.json') for dataset in datasets]
    val_file = [os.path.join(params.splits_dir, dataset, 'val.json') for dataset in datasets]
    train_few_shot_params = dict(n_way = params.train_n_way, n_support = params.n_shot)
    train_mixed_datamgr = SetDataManager(image_size, n_query = params.n_query, mixed_tasks = True,
                                            n_episode = params.mixed_task_batch_size, **train_few_shot_params)
    train_mixed_loader = train_mixed_datamgr.get_data_loader(params.data_dir, base_file, aug = params.train_aug)
    test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot)
    val_mixed_datamgr = SetDataManager(image_size, n_query = params.n_query, mixed_tasks = True,
                                        n_episode = params.mixed_val_batch_size, **test_few_shot_params)
    val_mixed_loader = val_mixed_datamgr.get_data_loader(params.data_dir, val_file, aug = params.train_aug)

    print("succesfully loaded mixed tasks!! :) \n")


    ## Create non domain-specific model (meta-learner) for mixed tasks ##
    use_pretrained_feature_extractor = True
    convnet.ConvBlock.maml = True ## new
    if params.task_network == 'linear':
        maml_convnet = None
    if params.task_network == 'conv1+linear' or params.task_network == 'conv2+linear' or params.task_network == 'conv3+linear':
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
        elif params.task_network == 'conv3+linear':
            print("using fixed pretrained feature extractor (conv1) for maml's task network (conv3 + linear layer) ")
            maml_convnet = convnet.ConvNet(depth = 3, preset_in_out_channels = conv1_channels,
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
                maml_convnet.final_feat_dim = 6400 ## obtained by outputting the shape of the features from the final conv layer

    model = MAML(pretrained_feature_extractor = feature_extractor,
                 model_info = (params.feature_extractor_model, params.task_network),
                 model_func = maml_convnet, **train_few_shot_params,
                 distance = params.distance_metric,
                 method = 'CosML', tf_path = params.tf_dir)
    model = model.cuda()


    ## Load domain-specific models and continue training (if necessary) ##
    if params.resume != '':
        print("/nresume training existing domain-specific meta-learners")
        resume_epochs = []

        for domain in seen_domains:
            dataset = domain.dataset
            resume_file, resume_epoch = get_resume_file('{}/{}/checkpoints/{}/{}'.format(params.save_dir, model_name, params.name, dataset), params.resume_epoch)
            print("resume file: {}".format(resume_file))
            resume_epochs.append(resume_epoch)
            if resume_file is not None:
                tmp = torch.load(resume_file)
                domain.model.load_state_dict(tmp['state'])
                domain.prototypes = tmp['prototypes']
        params.resume_epoch = np.min(resume_epoch)

    if params.resume_epoch >= params.start_epoch:
        start_epoch = params.resume_epoch + 1
    else:
        start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch


    ## Writing training configurations ##
    training_config_file = '{}/config.txt'.format(params.tf_dir)
    print("Saving training configurations to: {}\n".format(training_config_file))
    with open(training_config_file, 'w') as f:
        f.write("Training configurations - {}\n\n".format(params.name))
        f.write("data_dir: {}\n".format(params.data_dir))
        f.write("save_dir: {}\n".format(params.save_dir))
        f.write("splits_dir: {}\n".format(params.splits_dir))
        f.write("training dataset(s): {}\n".format(datasets))
        f.write("test set: {}\n".format(params.testset))
        f.write("model name: {}\n".format(params.name))
        f.write("method: {}\n".format(params.method))
        f.write("model: {}\n".format(params.model))
        f.write("init_num_channels: {}\n".format(params.init_num_channels))
        f.write("fix_num_channels: {}\n".format(params.fix_num_channels))
        f.write("train_type: {}\n".format(params.train_type))
        f.write("train_n_way: {}\n".format(params.train_n_way))
        f.write("n_shot: {}\n".format(params.n_shot))
        f.write("\nmixed_task_batch_size: {}\n".format(params.mixed_task_batch_size))
        f.write("pure_task_batch_size: {}\n".format(params.pure_task_batch_size))
        f.write("mixed_val_batch_size: {}\n".format(params.mixed_val_batch_size))
        f.write("pure_val_batch_size: {}\n\n".format(params.pure_val_batch_size))
        f.write("train_aug: {}\n".format(params.train_aug))
        f.write("save_freq: {}\n".format(params.save_freq))
        f.write("start_epoch: {}\n".format(params.start_epoch))
        f.write("resume_epoch: {}\n".format(params.resume_epoch))
        f.write("resume: {}\n".format(params.resume))
        f.write("pretrained_feature_extractor: {}\n".format(params.pretrained_feature_extractor))
        f.write("mixed_task_training: {}\n".format(params.mixed_task_training))


    training_log_file = os.path.join(params.tf_dir, 'training_log.txt')
    with open(training_log_file, 'a') as f:
        date_time = datetime.fromtimestamp(time.time())
        d = date_time.strftime("%c")
        f.write('\n\n---------------------- {} -------------------------\n'.format(d))




    ## Train ##
    data_loaders = dict(mixed_base_loader = train_mixed_loader,
                        mixed_val_loader = val_mixed_loader)
    train(log_file = training_log_file, model = model, seen_domains = seen_domains, **data_loaders,
            start_epoch = start_epoch, stop_epoch = stop_epoch, params = params)
