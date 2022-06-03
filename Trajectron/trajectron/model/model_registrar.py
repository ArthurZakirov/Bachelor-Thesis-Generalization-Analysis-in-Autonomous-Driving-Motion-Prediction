import os
import torch
import torch.nn as nn
import pathlib


def get_model_device(model):
    return next(model.parameters()).device


class ModelRegistrar(nn.Module):
    def __init__(self, model_dir, device):
        super(ModelRegistrar, self).__init__()
        self.model_dict = nn.ModuleDict()
        self.model_dir = model_dir
        self.device = device

    def forward(self):
        raise NotImplementedError('Although ModelRegistrar is a nn.Module, it is only to store parameters.')

    def get_model(self, name, model_if_absent=None):
        # 4 cases: name in self.model_dict and model_if_absent is None         (OK)
        #          name in self.model_dict and model_if_absent is not None     (OK)
        #          name not in self.model_dict and model_if_absent is not None (OK)
        #          name not in self.model_dict and model_if_absent is None     (NOT OK)

        if name in self.model_dict:
            return self.model_dict[name]

        elif model_if_absent is not None:
            self.model_dict[name] = model_if_absent.to(self.device)
            return self.model_dict[name]

        else:
            raise ValueError(f'{name} was never initialized in this Registrar!')

    def get_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    def get_all_but_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name not in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    def print_model_names(self):
        print(self.model_dict.keys())

    def save_models(self, curr_iter):
        # Create the model directiory if it's not present.
        save_path = os.path.join(self.model_dir,
                                 'model_registrar-%d.pt' % curr_iter)

        torch.save(self.model_dict, save_path)

    def save_optimizer_and_scheduler(self, optimizer_dict, scheduler_dict, curr_iter):
        optimizer_dir = os.path.join(self.model_dir, 'optimizer')
        scheduler_dir = os.path.join(self.model_dir, 'scheduler')
        pathlib.Path(optimizer_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(scheduler_dir).mkdir(parents=True, exist_ok=True)
        torch.save(optimizer_dict, os.path.join(optimizer_dir, 'optimizer-%d.pt' % curr_iter))
        torch.save(scheduler_dict, os.path.join(scheduler_dir, 'scheduler-%d.pt' % curr_iter))

    def load_models(self, iter_num):
        self.model_dict.clear()
        
        save_path = os.path.join(self.model_dir,
                                 'model_registrar-%d.pt' % iter_num)

        print('')
        print('Loading from ' + save_path)
        self.model_dict = torch.load(save_path, map_location=self.device)
        print('Loaded!')
        print('')

    def load_optimizer_and_scheduler_dict(self):
        max_checkpoint = 0
        for file in os.listdir(os.path.join(self.model_dir, 'optimizer')):
            checkpoint = int(file.split('.')[0].split('-')[1])
            if checkpoint > max_checkpoint:
                max_checkpoint = checkpoint
        optimizer_path = os.path.join(self.model_dir, 'optimizer', f'optimizer-{max_checkpoint}.pt')
        scheduler_path = os.path.join(self.model_dir, 'scheduler', f'scheduler-{max_checkpoint}.pt')
        optimizer_dict = torch.load(optimizer_path)
        scheduler_dict = torch.load(scheduler_path)
        return optimizer_dict, scheduler_dict

    def to(self, device):
        for name, model in self.model_dict.items():
            if get_model_device(model) != device:
                model.to(device)
