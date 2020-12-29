import numpy as np
import torch
import torch.optim
import os
import torchvision
import torch.nn.functional as F
from maml.models.conv_net import ConvModel
from maml.models.conv_net import ConvNet, PreTrain
from options import parse_args, get_resume_file, load_warmup_state
from data.pretrain.data_loader import PretrainDataset
from data.datamgr import SimpleDataManager
from tensorboardX import SummaryWriter
import time
from datetime import datetime
import json

import configs

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



def train(log_file, base_loader, val_loader, model, start_epoch, stop_epoch, params, device):


    # get optimizer and checkpoint path
    optimizer = torch.optim.Adam(model.parameters())
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # for validation
    max_acc = 0
    total_it = 0

    # model.cuda()

    print("in train(): starting training epochs...\n")
    # start
    for epoch in range(start_epoch,stop_epoch):
        epoch_start_time = time.time()
        date_time = datetime.fromtimestamp(epoch_start_time)
        d = date_time.strftime("%c")
        print("Epoch {}/{} - start_time: {}".format(epoch, stop_epoch-1, d))
        model.train()
        total_it, train_loss, train_acc = model.train_loop(epoch,
                                                            base_loader,
                                                            optimizer,
                                                            total_it,
                                                            device)

        model.eval()
        val_loss, val_acc = model.test_loop( val_loader, 'val')

        model.tf_writer.add_scalar('train/epoch_train_loss', train_loss, epoch+1)
        model.tf_writer.add_scalar('train/epoch_train_acc', train_acc, epoch+1)
        model.tf_writer.add_scalar('val/epoch_val_loss', val_loss, epoch+1)
        model.tf_writer.add_scalar('val/epoch_val_acc', val_acc, epoch+1)

        print("Epoch {:d} | Train Acc: {:f} | Train Loss: {:f} | Val Acc: {:f} | Val Loss: {:f}".format(epoch, train_acc, train_loss, val_acc, val_loss))

        with open(log_file, 'a') as f:
            f.write("Epoch {:d} | Train Acc: {:f} | Train Loss: {:f} | Val Acc: {:f} | Val Loss: {:f}\n".format(epoch, train_acc, train_loss, val_acc, val_loss))

        if val_acc > max_acc :
          print("best model! save...")
          max_acc = val_acc
          outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
          torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        else:
          print("GG! best accuracy {:f}".format(max_acc))

        if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
          outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
          torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        print("-- epoch {} took {:2} min {:.2f} sec -- ".format(epoch, int(epoch_elapsed_time//60), epoch_elapsed_time%60))



    return model

if __name__ == '__main__':

    np.random.seed(42)

    params = parse_args('train')


    print('-- pre-training : {}--\n'.format(params.name))
    print("defaut params: \n{}\n\n".format(params))

    ## Updating Params ##
    params.name = configs.options['pretrain']['name']
    params.method = configs.options['pretrain']['method']
    params.train_type = configs.options['pretrain']['train_type']
    params.dataset = configs.options['pretrain']['dataset']
    params.model = configs.options['pretrain']['model']#'Conv4'#'ResNet18' # base model
    params.train_aug = configs.options['pretrain']['train_aug']


    datasets = params.dataset
    testset_ls = params.testset.split(',')
    for testset in testset_ls:
        datasets.remove(testset)
    print("pretraining on: {}".format(datasets))
    print("test dataset: {}".format(params.testset))
    if params.train_type == 'nonepisodic':
        params.n_shot = 0
        all_classes = []
        datasets_name = '_'.join(datasets)
        with open(os.path.join(params.splits_dir, '{}_pretrain_train.json'.format(datasets_name)), 'r') as f:
            json_data = json.load(f)
        all_classes = json_data['label_names']
        params.num_classes = len(all_classes)
        print("params.num_classes: {}".format(params.num_classes))

    print("updated params: \n{}\n\n".format(params))


    params.init_num_channels = 64 #32
    params.fix_num_channels = True # False

    # output and tensorboard dir
    datasets_name = '_'.join(datasets)
    tensorboard_dir_name = '{}_{}'.format(params.name, datasets_name)
    if params.fix_num_channels:
        tensorboard_dir_name = tensorboard_dir_name + '_fixed_channels'
    tensorboard_dir_name = tensorboard_dir_name + '_init{}'.format(params.init_num_channels)
    if params.train_aug:
        tensorboard_dir_name = tensorboard_dir_name + '_trainaug'
    params.tf_dir = '{}/pretrain/log/{}'.format(params.save_dir, tensorboard_dir_name)
    params.checkpoint_dir = '{}/pretrain/checkpoints/{}'.format(params.save_dir, tensorboard_dir_name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    if not os.path.isdir(params.tf_dir):
        os.makedirs(params.tf_dir)



    # prepare dataloader
    print("\n preparing dataloader ... \n")
    params.train_batch_size = 128 #64
    params.val_batch_size = 64 # 64

    print("train batch size: {} | val batch size: {}".format(params.train_batch_size, params.val_batch_size) )
    train_file_name = "{}_pretrain_train.json".format(datasets_name)
    train_file_json = os.path.join(params.splits_dir, train_file_name)
    train_datamgr = SimpleDataManager(image_size = 84,
                                        batch_size = params.train_batch_size)
    train_dataloader = train_datamgr.get_data_loader(base_data_dir = params.data_dir,
                                                    # data_split = 'train',
                                                    data_file = train_file_json,
                                                    aug = params.train_aug)
    val_file_name = "{}_pretrain_val.json".format(datasets_name)
    val_file_json = os.path.join(params.splits_dir, val_file_name)
    val_datamgr = SimpleDataManager(image_size = 84,
                                        batch_size = params.val_batch_size)
    val_dataloader = train_datamgr.get_data_loader(base_data_dir = params.data_dir,
                                                    # data_split = 'val',
                                                    data_file = val_file_json,
                                                    aug = False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}\n\n".format(device))
    conv_net = ConvNet(depth = 4, init_num_channels = params.init_num_channels,
                        fix_num_channels = params.fix_num_channels, flatten = True)
    # if params.feature_extractor_model == 'conv4':
    if params.init_num_channels == 64 and params.fix_num_channels:
        conv_net.final_feat_dim = 1600
    elif params.init_num_channels == 32 and not params.fix_num_channels:
        ## for variable channels ##
        conv_net.final_feat_dim = 6400 ## obtained by outputting the shape of the features from the final conv layer

    print("succesfully initialized ConvModel")
    model = PreTrain(model_func = conv_net, num_class = params.num_classes, tf_path = params.tf_dir)
    model = model.cuda()







    ## Load model and continue training (if necessary) ##
    if params.resume != '':
        resume_file, resume_epoch = get_resume_file('{}/{}/checkpoints/{}'.format(params.save_dir, params.method, tensorboard_dir_name), params.resume_epoch)
        print("resume training from epoch {} ... ".format("resume_epoch"))
        params.resume_epoch = resume_epoch
        print("resume file: {}\n".format(resume_file))
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
            print("\tresume the training of {} at {} epoch".format(tensorboard_dir_name, start_epoch))

    if params.resume_epoch >= params.start_epoch:
        start_epoch = params.resume_epoch + 1
        params.start_epoch = start_epoch
    else:
        start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    print("start_epoch: {}".format(start_epoch))
    print("stop_epoch: {}".format(stop_epoch))


    ## Writing training configurations ##
    with open("{}/config.txt".format(params.tf_dir), 'w') as f:
        f.write("training configurations\n\n")
        f.write("data_dir: {}\n".format(params.data_dir))
        f.write("dataset: {}\n".format(params.dataset))
        f.write("method: {}\n".format(params.method))
        f.write("model: {}\n".format(params.model))
        f.write("train_type: {}\n".format(params.train_type))
        f.write("num_classes: {}\n".format(params.num_classes))
        f.write("save_dir: {}\n".format(params.save_dir))
        f.write("name: {}\n".format(params.name))
        f.write("save_freq: {}\n".format(params.save_freq))
        f.write("start_epoch: {}\n".format(params.start_epoch))
        f.write("stop_epoch: {}\n".format(params.stop_epoch))
        f.write("resume: {}\n".format(params.resume))
        f.write("resume_epoch: {}\n".format(params.resume_epoch))
        f.write("train_aug: {}\n".format(params.train_aug))
        f.write("train_batch_size: {}\n".format(params.train_batch_size))
        f.write("val_batch_size: {}\n".format(params.val_batch_size))
        f.write("init_num_channels: {}\n".format(params.init_num_channels))
        f.write("fix_num_channels: {}\n".format(params.fix_num_channels))

    training_log_file = os.path.join(params.tf_dir, 'training_log.txt')
    with open(training_log_file, 'a') as f:
        f.write('\n-------------------------------------------------\n\n')




    ## Training ##
    print("\n-- training begins -- ")
    model = train(  log_file = training_log_file,
                    base_loader = train_dataloader,
                    val_loader = val_dataloader,
                    model = model,
                    start_epoch = start_epoch,
                    stop_epoch = stop_epoch,
                    params=params,
                    device = device)
    print("\n-- training completed! :) --\n\n")
