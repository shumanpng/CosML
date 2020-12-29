# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml

# import backbone
import maml.models.conv_net as convnet
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.utils import euclidean_dist
import math
import copy

class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, distance = 'euclidean', model_info = None, pretrained_feature_extractor = None, method = None, tf_path = None, approx = False, device='cuda:0'):
        super(MAML, self).__init__( model_func, n_way, n_support, model_info, pretrained_feature_extractor, change_way = False, tf_path = tf_path)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = convnet.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        self.pretrained_features = True if pretrained_feature_extractor is not None else False
        self.has_fixed_layers = True # true by default
        self.device = device

        self.n_task     = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.optimizer_lr = 0.001 # default Adam learning rate is 0.001
        self.approx = approx #first order approx.
        self.distance_metric = distance
        self.adapt_feature_extractor = False
        # self.cos_sim = torch.nn.CosineSimilarity(dim = 1, eps=1e-08)

        if method is None:
            self.method = 'MAML'
        else:
            self.method = method
        print("MAML.classifier: {}".format(self.classifier))
        print("MAML.distance_metric: {}\n".format(self.distance_metric))


    def get_intermediate_features(self, x):
        """
        return:
            - inter_x: intermediate feature representation of original task x
            - intermediate_features:  a list of intermediate features of each example in task x

        """
        x = x.to(self.device)
        intermediate_features = []

        ## get intermediate feature of one data point at a time and then batch it back into
        ## one task ##
        if self.pretrained_features:
            inter_x = torch.Tensor() # init     intermediate feature representation of x (task)
            img_count = 0
            for cls in range(x.shape[0]):
                inter_imgs = self.intermediate_feature.forward(x[cls])
                if cls == 0:
                    inter_x = inter_imgs
                else:
                    inter_x = torch.cat([inter_x, inter_imgs])
                inter_imgs_flatten = inter_imgs.view(inter_imgs.size(0), -1)
                intermediate_features.append(inter_imgs_flatten)
                for ind, img in enumerate(x[cls]):
                    img_count += 1
            reshape = list(x.shape) # x.shape: torch.Size([5, 21, 3, 84, 84])
            reshape[2:] = inter_x.shape[1:] # reshape: [5, 21, 128, 10, 10]
            inter_x = inter_x.view(reshape) # reshape to original x shape in terms of # way and batch size (support + query)
        return inter_x, intermediate_features



    def forward(self,x):
        if self.feature is not None:
            out  = self.feature.forward(x)
            scores  = self.classifier.forward(out)
        else:
            ## flatten features ##
            x = x.view(x.size(0), -1)
            scores = self.classifier.forward(x)
        return scores

    def set_forward(self,x, is_feature = False):
        """

        return:
            - scores
        """
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.to(self.device)

        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).to(self.device)

        require_grad_params = filter(lambda param: param.requires_grad, self.parameters())

        fast_parameters = list(require_grad_params)
        for i, weight in enumerate(fast_parameters):
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i)
            if self.pretrained_features and self.has_fixed_layers:
                grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused = True)
                ## hack to fix CUDA memory issue ##
                # grad = torch.autograd.grad(set_loss, fast_parameters, allow_unused = True)
            else:
                ## ORIGINAL IMPLEMENTATION:
                grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
                ## hack to fix CUDA memory issue: don't create graphs for gradient update steps ##
                # grad = torch.autograd.grad(set_loss, fast_parameters)
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            require_grad_params = filter(lambda param: param.requires_grad, self.parameters()) ## me


            for k, weight in enumerate(require_grad_params):
                if grad[k] is None:
                    weight.fast = weight
                if grad[k] is not None:
                    if weight.fast is None:
                        weight.fast = weight - self.train_lr * grad[k] #create weight.fast
                    else:
                        weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(weight.fast)
        require_grad_params = list(filter(lambda param: param.requires_grad, self.parameters()))
        scores = self.forward(x_b_i)

        ## try del variable to save CUDA memory ##
        del x_var, y_a_i
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).to(self.device)
        loss = self.loss_fn(scores, y_b_i)

        ## try del variable to save cuda memory ##
        del y_b_i
        return scores, loss

    def train_loop(self, epoch, train_loader, optimizer, total_it, dataset = None, compute_prototypes = False): #overwrite parent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        prototypes = {}
        if compute_prototypes:
            prototypes['domain'] = {'count': 0, 'mean': None}
            prototypes['task'] = []

        all_task_features = [] # total features of all tasks
        #train
        for i, (x,_) in enumerate(train_loader): # x is a task
            task_features = [] # features of one task
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            if self.pretrained_features:
                inter_x, intermediate_features = self.get_intermediate_features(x)
                x = inter_x

            _, loss = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()#data[0]
            loss_all.append(loss)

            if compute_prototypes:
                task_features = torch.cat(intermediate_features)
                all_task_features.append(task_features)
                task_proto = task_features.mean(dim=0)
                prototypes['task'].append(task_proto)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar('/pure_task/{}/query_loss'.format(dataset), loss.item(), total_it + 1)

            total_it += 1

        if compute_prototypes:
            ## update domain prototype and count ##
            domain_features = torch.cat(all_task_features)
            domain_proto = domain_features.mean(dim=0)
            prototypes['domain']['count'] = domain_features.size(0) # total number of imgs (features) in this train_loop call
            prototypes['domain']['mean'] = domain_proto
        return total_it, prototypes


    def update_params(self, domain, loss, last_call = False):
        self.zero_grad()
        require_grad_params = filter(lambda param: param.requires_grad, self.parameters()) ## me
        require_grad_params = list(require_grad_params)

        grad = torch.autograd.grad(loss, require_grad_params, create_graph = True, allow_unused = True)
        k = 0
        for weight in domain.model.parameters():
            if weight.requires_grad:
                weight.fast = None

                if grad[k] is None:
                    print("grad[{}] is None!!".format(k))
                if grad[k] is not None:
                    weight.data -= self.optimizer_lr*grad[k].data
                k += 1


    def mixed_train_loop(self, epoch, train_loader, optimizer, total_it, compute_prototypes = False, seen_domains = None, has_prototypes = True, uniform_weight = False): #overwrite parrent function
        """
        params
            epoch: current training epoch
            train_loader: dataloader for mixed tasks
            optimizer: mixed task model optimizer
            total_it: running count of the total number of iterations done for mixed task training
                        (i.e., total number of mixed tasks used for training)
            compute_prototypes: whether or not domain + task prototypes should be computed
            seen_domains: list of Domain objects for each seen domain
                            Domain object's attributes:
                                - dataset (name of dataset),
                                - prototypes ({'domain': {'count': - , 'mean': tensor[]}
                                                'task': [])
                                - model
                                - optimizer
                                - train_dataloader
                                - val_dataloader
                                - test_dataloader
                                - total_it


            ** note:
                if has_prototypes == False then uniform_weight MUST BE TRUE!
        return:
            total_it (of mixed task training)


        """
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()
        self.zero_grad() ##
        domain_loss_all = {domain.dataset: [] for domain in seen_domains}
        cos = torch.nn.CosineSimilarity(dim = 1, eps=1e-08)
        num_seen_domains = len(seen_domains) if seen_domains is not None else -1

        all_task_features = [] # total features of all tasks
        #train
        for i, (x,_) in enumerate(train_loader): # x is a task
            task_features = [] # features of one task
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            if self.pretrained_features:
                ## Step 1: compute intermediate features ##
                inter_x, intermediate_features = self.get_intermediate_features(x)
                x = inter_x

                if has_prototypes:
                    ## compute euclidean dist between each intermediate feature for each example in
                    ## task x in intermediate_features to domain + task prototypes of each seen domain
                    domain_weights = [] ## alpha_Di
                    domain_distances = []
                    ## Step 2: compute each seen domain's mixture weight for this task ##
                    for domain in seen_domains:

                        stacked_intermediate_features = torch.stack(intermediate_features)
                        support_inter_features = stacked_intermediate_features[:, :self.n_support, :] # shape [5, 5, 12800] -> [n_way, n_support, dim]
                        support_inter_features = support_inter_features.contiguous().view(self.n_way*self.n_support, -1) # shape [25, 12800]

                        ## Compute distance from the support set of this task to the domain prototype ##
                        domain_proto = domain.prototypes['domain']['mean'].contiguous().view(1, -1) # change from size [12800] to [1, 12800] for euclidean_dist
                        if self.distance_metric == 'euclidean':
                            dists_to_dp = euclidean_dist(support_inter_features, domain_proto) ## Euclidean distance ##
                        if self.distance_metric == 'cosine':
                            dists_to_dp = cos(support_inter_features, domain_proto)
                            ones = dists_to_dp.clone()
                            ones.data.fill_(1)
                            dists_to_dp = ones - dists_to_dp # cosine distance = 1 - cosine similarity

                        mean_dp_dist = dists_to_dp.mean(0).cpu().numpy()

                        if self.distance_metric == 'euclidean':
                            mean_tp_dist_ls = []
                            task_proto_batch_size = 300 ## to prevent CUDA out of memory issue
                            num_task_protos = len(domain.prototypes['task'])
                            if num_task_protos > task_proto_batch_size:
                                ## split task prototypes up into batches to Euclidean distance computation ##
                                num_batches = math.ceil(num_task_protos/task_proto_batch_size)
                                for batch_idx in range(num_batches):
                                    if batch_idx == 0:
                                        task_proto_batch = torch.stack(domain.prototypes['task'][:task_proto_batch_size])
                                    elif batch_idx == num_batches - 1:
                                        task_proto_batch = torch.stack(domain.prototypes['task'][batch_idx * task_proto_batch_size: ])
                                    else:
                                        start_idx = batch_idx * task_proto_batch_size
                                        end_idx = (batch_idx * task_proto_batch_size) + task_proto_batch_size
                                        task_proto_batch = torch.stack(domain.prototypes['task'][start_idx:end_idx])
                                    dists_to_tp1 = euclidean_dist(support_inter_features, task_proto_batch)
                                    mean_tp_dist_ls.extend(dists_to_tp1.mean(0).cpu().numpy())
                                    mean_tp_dists = mean_tp_dist_ls
                            else:
                                task_protos = torch.stack(domain.prototypes['task'])    # shape [# of task protos, 12800]
                                dists_to_tp = euclidean_dist(support_inter_features, task_protos) # shape: [n_way*n_support, # of task protos]
                                mean_tp_dists = dists_to_tp.mean(0).cpu().numpy() # shape: [# of task protos]

                        if self.distance_metric == 'cosine':
                            cossim_to_each_proto = []
                            for tp in domain.prototypes['task']:
                                tp = tp.contiguous().view(1, -1)
                                tp_cossim = cos(support_inter_features, tp)
                                cossim_to_each_proto.append(tp_cossim)
                            dists_to_tp = torch.stack(cossim_to_each_proto)
                            ones = dists_to_tp.clone()
                            ones.data.fill_(1)
                            dists_to_tp = ones - dists_to_tp # cosine distance = 1 - cosine similarity
                            mean_tp_dists = dists_to_tp.mean(1).cpu().numpy()

                        distance = (1/2) * (mean_dp_dist.item() + np.mean(mean_tp_dists))
                        dom_weight = 1/distance
                        domain.mix_weight = dom_weight
                        domain_weights.append(dom_weight)

            ## normalize domain.mix_weight ##
            for domain in seen_domains:
                if uniform_weight:
                    domain.mix_weight = 1.0 / float(num_seen_domains)
                else:
                    domain.mix_weight = domain.mix_weight / np.sum(domain_weights) # normalize domain mix weight


            ## Step 3: initialize the params of the (mixed) task network ##
            for k, param in enumerate(self.named_parameters()):
                if param[1].requires_grad:
                    param[1].data.fill_(0)

            require_grad_params = filter(lambda param: param.requires_grad, self.parameters() )
            require_grad_params = list(require_grad_params)
            for k, weight in enumerate(require_grad_params):
                if weight.requires_grad:
                    for domain in seen_domains:
                        domain_params = list(filter(lambda param: param.requires_grad, domain.model.parameters()))
                        weight.data += domain_params[k].data * domain.mix_weight


            ## Step 4: compute mixed task loss on query set using adapted task network ##
            _, loss = self.set_forward_loss(x)

            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            ## multiply loss with domain mixture weight before adding it to domain-specific
            ## loss all list for later use (to update domain specific params after meta batch-size/n_task
            ## number of mixed tasks )
            for domain in seen_domains:
                weighted_loss = loss * domain.mix_weight ## COMMENT MIXWEIGHT BACK IN LATER AFTER DEBUGGING!
                domain_loss_all[domain.dataset].append(weighted_loss)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                last_call = False
                for d_idx, domain in enumerate(seen_domains):

                    ## Step 5.1: sum up weighted loss_all for each domain ##
                    loss_q = torch.stack(domain_loss_all[domain.dataset]).sum(0)
                    if d_idx == len(seen_domains) - 1:
                        last_call = True
                    domain.model.train()
                    self.update_params(domain, loss_q, last_call = last_call) # no need to return because domain params are updated in this function
                    domain.model.eval()
                    domain_loss_all[domain.dataset] = []

                task_count = 0
                loss_all = []

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar('/mixed_task/query_loss', loss.item(), total_it + 1)

            total_it += 1

        return total_it


    def test_loop(self, test_loader, return_std = False, verbose=True): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        loss = 0.

        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            if self.pretrained_features:
                inter_x, intermediate_features = self.get_intermediate_features(x)
                x = inter_x

            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )
            loss += loss_this
            count += count_this

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        loss_mean = loss/count
        if verbose:
            print('%d Loss = %.6f'%(iter_num, loss_mean))
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std, acc_all
        else:
            return acc_mean, loss_mean, acc_all


    def mixed_test_loop(self, test_loader, seen_domains, return_std = False, verbose = True, test = False, has_prototypes = True, uniform_weight = False):
        correct = 0
        count = 0
        acc_all = []
        loss = 0.
        cos = torch.nn.CosineSimilarity(dim = 1, eps=1e-08)
        num_seen_domains = len(seen_domains)

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"

            ## initialize model parameters using domain specific meta-learners ##
            if self.pretrained_features:

                inter_x, intermediate_features = self.get_intermediate_features(x)
                x = inter_x

                if has_prototypes:
                    domain_weights = []
                    domain_distances = []
                    for domain in seen_domains:
                        if domain.prototypes['domain']['mean'] is None:
                            continue
                        stacked_intermediate_features = torch.stack(intermediate_features)
                        support_inter_features = stacked_intermediate_features[:, :self.n_support, :]
                        support_inter_features = support_inter_features.contiguous().view(self.n_way*self.n_support, -1)

                        domain_proto = domain.prototypes['domain']['mean'].contiguous().view(1, -1)
                        if self.distance_metric == 'euclidean':
                            dists_to_dp = euclidean_dist(support_inter_features, domain_proto)
                        elif self.distance_metric == 'cosine':
                            dists_to_dp = cos(support_inter_features, domain_proto)
                            ones = dists_to_dp.clone()
                            ones.data.fill_(1)
                            dists_to_dp = ones - dists_to_dp # cosine distance = 1 - cosine similarity
                        mean_dp_dist = dists_to_dp.mean(0).cpu().detach().numpy()

                        if self.distance_metric == 'euclidean':
                            mean_tp_dist_ls = []
                            task_proto_batch_size = 300 ## to prevent CUDA out of memory issue
                            num_task_protos = len(domain.prototypes['task'])
                            if num_task_protos > task_proto_batch_size:
                                ## split task prototypes up into batches to Euclidean distance computation ##
                                num_batches = math.ceil(num_task_protos/task_proto_batch_size)
                                for batch_idx in range(num_batches):
                                    if batch_idx == 0:
                                        task_proto_batch = torch.stack(domain.prototypes['task'][:task_proto_batch_size])
                                    elif batch_idx == num_batches - 1:
                                        task_proto_batch = torch.stack(domain.prototypes['task'][batch_idx * task_proto_batch_size: ])
                                    else:
                                        start_idx = batch_idx * task_proto_batch_size
                                        end_idx = (batch_idx * task_proto_batch_size) + task_proto_batch_size
                                        task_proto_batch = torch.stack(domain.prototypes['task'][start_idx:end_idx])
                                    dists_to_tp1 = euclidean_dist(support_inter_features, task_proto_batch)
                                    mean_tp_dist_ls.extend(dists_to_tp1.mean(0).cpu().detach().numpy())
                                    mean_tp_dists = mean_tp_dist_ls
                            else:
                                task_protos = torch.stack(domain.prototypes['task'])
                                dists_to_tp = euclidean_dist(support_inter_features, task_protos)
                                mean_tp_dists = dists_to_tp.mean(0).cpu().detach().numpy()

                        elif self.distance_metric == 'cosine':
                            cossim_to_each_proto = []
                            for tp in domain.prototypes['task']:
                                tp = tp.contiguous().view(1, -1)
                                tp_cossim = cos(support_inter_features, tp)
                                cossim_to_each_proto.append(tp_cossim)
                            dists_to_tp = torch.stack(cossim_to_each_proto)
                            ones = dists_to_tp.clone()
                            ones.data.fill_(1)
                            dists_to_tp = ones - dists_to_tp # cosine distance = 1 - cosine similarity
                            mean_tp_dists = dists_to_tp.mean(1).cpu().numpy()

                        distance = (1/2) * (mean_dp_dist.item() + np.mean(mean_tp_dists))
                        dom_weight = 1/distance
                        domain.mix_weight = dom_weight
                        domain_weights.append(dom_weight)

            # normalize weights #
            for domain in seen_domains:
                if uniform_weight:
                    domain.mix_weight = 1.0/float(num_seen_domains)
                else:
                    domain.mix_weight = domain.mix_weight / np.sum(domain_weights)

            ## initialize task network for this mixed validation task ##
            for k, param in enumerate(self.named_parameters()):
                if param[1].requires_grad:
                    param[1].data.fill_(0)

            require_grad_params = filter(lambda param: param.requires_grad, self.parameters())
            require_grad_params = list(require_grad_params)
            for k, weight in enumerate(require_grad_params):
                if weight.requires_grad:
                    for domain in seen_domains:
                        domain_params = list(filter(lambda param: param.requires_grad, domain.model.parameters()))
                        weight.data += domain_params[k].data * domain.mix_weight
            self.eval()
            x = x.to(self.device)
            ## end of task network weight initialization ##

            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this/count_this * 100)
            loss += loss_this
            count += count_this

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        loss_mean = loss/count
        if verbose:
            print('%d Loss = %.6f'%(iter_num, loss_mean))
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std, acc_all
        else:
            return acc_mean, loss_mean, acc_all
