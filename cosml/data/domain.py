

class Domain():
    def __init__(self, dataset, prototypes = {}, model = None, optimizer = None, tr_dataloader = None, val_dataloader = None, test_dataloader = None):
        '''
        dataset: name of the dataset (e.g., 'miniimagenet')
        '''
        self.dataset = dataset
        # self.prototypes = prototypes
        # if 'domain' not in prototypes:
        #     ## count and mean helps us recover the original total,
        #     ## which we need when we are updating the domain prototype
        #     ## using new batch tasks' data
        #     self.prototypes['domain'] = {'count': 0, 'mean': None}
        # if 'class' not in prototypes:
        #     ## tbh probably don't need this
        #     self.prototypes['class'] = []
        # if 'task' not in prototypes:
        #     ## just a list of mean feature representations for each
        #     ## task will do because we are assuming that no two tasks
        #     ## will be the same
        #     self.prototypes['task'] = []
        self.prototypes = {'domain': {'count': 0, 'mean': None},
                            'task': []}
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = tr_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # running count of the total number of training iterations done for this domain
        # 1 iteration == 1 training task
        self.total_it = 0
        self.mix_weight = 0
        self.mix_loss = None
        self.max_acc = 0

    def update_params(self):
        self.optimizer.zero_grad()
        # loss_cp.backward()
        # if last_call:
        self.mix_loss.backward()
        # else:
            # self.loss.backward(retain_graph = True)
        # domain.mix_loss.backward()
        self.optimizer.step()
