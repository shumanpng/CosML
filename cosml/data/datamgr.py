# This code is from https://github.com/hytseng0509/CrossDomainFewShot/blob/master/data/datamgr.py

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, MultiSetDataset, EpisodicBatchSampler, MultiEpisodicBatchSampler, MultiEpisodicMixedBatchSampler
from abc import abstractmethod

class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='RandomResizedCrop':
      return method(self.image_size)
    elif transform_type=='CenterCrop':
      return method(self.image_size)
    elif transform_type=='Resize':
      return method([int(self.image_size*1.15), int(self.image_size*1.15)])
    elif transform_type=='Normalize':
      return method(**self.normalize_param )
    else:
      return method()

  def get_composed_transform(self, aug = False):
    if aug:
      transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

    transform_funcs = [ self.parse_transform(x) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    return transform

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, base_data_dir, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(base_data_dir, data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader

class SetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, mixed_tasks = False, n_episode=100):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_episode = n_episode
    self.mixed_tasks = mixed_tasks

    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, base_data_dir, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      if self.mixed_tasks:
          print("creating mixed task sampler...\n")
          dataset = MultiSetDataset( base_data_dir, data_file, self.batch_size, transform)
          sampler = MultiEpisodicMixedBatchSampler(dataset.lens(), self.n_way, self.n_episode)
      else:
          dataset = MultiSetDataset( base_data_dir, data_file , self.batch_size, transform )
          sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_episode )
    else:
      dataset = SetDataset( base_data_dir, data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    # print("\n\ndatamgr SetDatasetManager.get_data_loader: dataset {}\n\n".format(dataset))
    return data_loader
