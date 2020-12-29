## adapted from https://github.com/hytseng0509/CrossDomainFewShot/blob/master/options.py

import numpy as np
import os
import glob
import torch
import argparse

def parse_args(script):
  parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
  parser.add_argument('--dataset', default='multi', help='miniImagenet/cub/cars/places/plantae, specify multi for training with multiple domains')
  parser.add_argument('--testset', default='cub', help='cub/cars/places/plantae, valid only when dataset=multi')
  parser.add_argument('--model', default='ResNet10', help='model: Conv{4|6} / ResNet{10|18|34}')
  parser.add_argument('--method', default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/gnnnet')
  parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
  parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
  parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
  parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
  parser.add_argument('--name'        , default='tmp', type=str, help='')
  parser.add_argument('--save_dir'    , default='./output', type=str, help='')
  parser.add_argument('--data_dir'    , default='../data_loader/data/', type=str, help='')
  parser.add_argument('--train_type'  , default='episodic', type=str, help='nonepisodic/episodic')
  parser.add_argument('--pretrained_feature_extractor', default='', type=str, help='path of the saved model')
  parser.add_argument('--splits_dir', default='../data/crossdomain_data/', type=str, help='')
  parser.add_argument('--feature_extractor_model', default='conv4', type=str, help='conv4/resnet10')
  parser.add_argument('--task_network', default='conv1+linear', type=str, help='conv1+linear/conv2+linear/conv3+linear/linear')
  parser.add_argument('--distance_metric', default='euclidean', type=str, help='cosine/euclidean')
  parser.add_argument('--meta_learner_path', default='', type=str, help='direct path to trained meta-learner')
  parser.add_argument('--mixed_task_batch_size', default=25, type=int, help='# of mixed tasks in a training batch')
  parser.add_argument('--pure_task_batch_size', default=25, type=int, help='# of pure tasks in a training batch')
  parser.add_argument('--mixed_val_batch_size', default=10, type=int, help='')
  parser.add_argument('--pure_val_batch_size', default=10, type=int, help='')
  parser.add_argument('--warmup', default = '', type=str, help='')
  parser.add_argument('--warmup_epoch', default=0, type=int, help='')
  parser.add_argument('--mixed_task_training', default='yes', type=str, help='whether or not to perform training with mixed tasks ')
  parser.add_argument('--mix_models', default='', type=str, help='uniform/adaptive')
  parser.add_argument('--gconvmaml', default='no', type=str, help='if yes, then the feature extractor would also use Conv2d_fw and BatchNorm2d_fw')
  parser.add_argument('--num_classes' , default=100, type=int, help='total number of classes in softmax, only used in baseline')
  if script == 'train':
    parser.add_argument('--save_freq'   , default=25, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch')
    parser.add_argument('--resume'      , default='', type=str, help='continue from previous trained model with largest epoch')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='')
  elif script == 'test':
    parser.add_argument('--split'       , default='novel', help='base/val/novel')
    parser.add_argument('--save_epoch', default=399, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
  else:
    raise ValueError('Unknown script')

  return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
  assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
  return assign_file

def get_assigned_file_exact_path(full_model_path):
  assign_file = os.path.join(full_model_path)
  return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
  filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
  if len(filelist) == 0:
    return None

  filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
  epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
  max_epoch = np.max(epochs)
  epoch = max_epoch if resume_epoch == -1 else resume_epoch
  resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
  return resume_file, epoch

def get_best_file(checkpoint_dir):
  best_file = os.path.join(checkpoint_dir, 'best_model.tar')
  if os.path.isfile(best_file):
    return best_file
  else:
    return get_resume_file(checkpoint_dir)

def load_warmup_state(filename, method, resume_epoch = -1):
  print('  load pre-trained model file: {}'.format(filename))
  warmup_resume_file, _ = get_resume_file(filename, resume_epoch)
  print("warmup_resume_file: {}\n".format(warmup_resume_file))
  tmp = torch.load(warmup_resume_file)
  if tmp is not None:
    state = tmp['state']
    state_keys = list(state.keys())
    print("(load_warmup_state) state_keys: {}\n".format(state_keys))
    for i, key in enumerate(state_keys):
      if 'relationnet' in method and "feature." in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif method == 'gnnnet' and 'feature.' in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)

      elif method == 'maml' and 'feature.' in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif method == 'protonet' and 'feature.' in key:
        newkey = key.replace("feature.", "")
        state[newkey] = state.pop(key)
      elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      else:
        state.pop(key)
  else:
    raise ValueError(' No pre-trained encoder file found!')
  return state
