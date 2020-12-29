# This code is from https://github.com/hytseng0509/CrossDomainFewShot/blob/master/data/dataset.py

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
identity = lambda x:x


class SimpleDataset:
  def __init__(self, base_data_dir, data_file, transform, target_transform=identity):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = transform
    self.target_transform = target_transform
    self.base_data_dir = base_data_dir
    # self.data_split = data_split

  def __getitem__(self,i):
    image_path = os.path.join(self.base_data_dir, self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])


class SetDataset:
  def __init__(self, base_data_dir, data_file, batch_size, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.cl_list = np.unique(self.meta['image_labels']).tolist()

    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []

    for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
      self.sub_meta[y].append(x)

    self.sub_dataloader = []
    sub_data_loader_params = dict(batch_size = batch_size,
        shuffle = True,
        num_workers = 0, #use main thread only or may receive multiple batches
        pin_memory = False)
    for cl in self.cl_list:
      sub_dataset = SubDataset(base_data_dir, self.sub_meta[cl], cl, transform = transform )
      self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)


class MultiSetDataset:
  def __init__(self, base_data_dir, data_files, batch_size, transform):
    self.base_data_dir = base_data_dir
    self.cl_list = np.array([])
    self.sub_dataloader = []
    self.n_classes = []
    for data_file in data_files:
      with open(data_file, 'r') as f:
        meta = json.load(f)
      cl_list = np.unique(meta['image_labels']).tolist()
      self.cl_list = np.concatenate((self.cl_list, cl_list))

      sub_meta = {}
      for cl in cl_list:
        sub_meta[cl] = []

      for x,y in zip(meta['image_names'], meta['image_labels']):
        sub_meta[y].append(x)

      sub_data_loader_params = dict(batch_size = batch_size,
          shuffle = True,
          num_workers = 0, #use main thread only or may receive multiple batches
          pin_memory = False)
      for cl in cl_list:
        sub_dataset = SubDataset(base_data_dir, sub_meta[cl], cl, transform = transform, min_size=batch_size)
        self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
      self.n_classes.append(len(cl_list))

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)

  def lens(self):
    return self.n_classes


class SubDataset:
  def __init__(self, base_data_dir, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
    self.base_data_dir = base_data_dir
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform
    self.target_transform = target_transform
    if len(self.sub_meta) < min_size:
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

  def __getitem__(self,i):
    image_path = os.path.join( self.base_data_dir, self.sub_meta[i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.cl)
    return img, target

  def __len__(self):
    return len(self.sub_meta)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_classes)[:self.n_way]


class MultiEpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes
    self.n_domains = len(n_classes)

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    domain_list = [i%self.n_domains for i in range(self.n_episodes)]
    random.shuffle(domain_list)
    for i in range(self.n_episodes):
      domain_idx = domain_list[i]
      start_idx = sum(self.n_classes[:domain_idx])
      torch_rand_perm = torch.randperm(self.n_classes[domain_idx])[:self.n_way]

      yield torch_rand_perm + start_idx

class MultiEpisodicMixedBatchSampler(object):
    """
    This sampler samples batches of mixed tasks :)
    """

    def __init__(self, n_classes, n_way, n_episodes):
        """
        n_classes: a list that contains the number of classes in each dataset
                    --> len(n_classes) == number of datasets
        n_way: # of classes in each task
        n_episodes: # of tasks in each episodic training batch
        """
        self.n_classes = n_classes # a list that contains the
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.n_domains = len(n_classes)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            selected_domains_idx = []   ## FOR DEBUGGING
            num_classes_per_domain = [] ## FOR DEBUGGING

            ## for each task ##
            domain_list = list(range(self.n_domains))
            random.shuffle(domain_list)
            ## randomly sample a # between [0, n_way//2 (inclusive!!)] for # of classes
            ## in the first dataset (in randomly shuffled order)
            num_classes = np.random.randint(1, self.n_way//2 + 1)
            selected_way = num_classes
            remaining_way = self.n_way - selected_way

            domain_idx = domain_list.pop(0) # pop first domain idx on list
            start_idx = sum(self.n_classes[:domain_idx])
            torch_rand_perm = torch.randperm(self.n_classes[domain_idx])[:num_classes] + start_idx
            task = torch_rand_perm

            selected_domains_idx.append(domain_idx) ## FOR DEBUGGING
            num_classes_per_domain.append(num_classes)     ## FOR DEBUGGING

            while len(domain_list) > 0:
                num_classes = np.random.randint(0, remaining_way+1)
                domain_idx = domain_list.pop(0)
                if len(domain_list) == 0:
                    # get all remaining classes from this last domain
                    num_classes = remaining_way

                if num_classes > 0:
                    selected_way += num_classes
                    remaining_way -= num_classes
                    start_idx = sum(self.n_classes[:domain_idx])
                    torch_rand_perm = torch.randperm(self.n_classes[domain_idx])[:num_classes] + start_idx
                    task = torch.cat((task, torch_rand_perm), 0)

                    selected_domains_idx.append(domain_idx) ## FOR DEBUGGING
                    num_classes_per_domain.append(num_classes)     ## FOR DEBUGGING

                if remaining_way == self.n_way:
                    break
            # print("domains (idx) that are represented in this task: {}".format(selected_domains_idx))
            # print("number of classes in each selected domain: {}".format(num_classes_per_domain))
            # print("n_classes: {}".format(self.n_classes))
            # print("task class indices: {}\n\n".format(task))

            yield task
