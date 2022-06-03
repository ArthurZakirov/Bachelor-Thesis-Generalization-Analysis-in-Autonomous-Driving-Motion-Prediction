from torch.utils import data
import numpy as np
from .preprocessing import get_node_timestep_data
import sys
sys.path.append('../data_analysis')
sys.path.append('../../Trajectron/data_analysis')
sys.path.append('../../experiments/Trajectron/data_analysis')

from scene_analysis import low_turning_angle


class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, data_dir, **kwargs):
        self.data_dir = data_dir
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = []
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, self._augment, data_dir, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment, data_dir, **kwargs):
        self.data_dir = data_dir
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = False

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]


    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = []
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            straight = 0
            turn = 0
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    if 'openDD' in self.data_dir and node_freq_mult:
                        if low_turning_angle(node):
                            node.frequency_multiplier = 20
                            straight += 1
                        else:
                            node.frequency_multiplier = 1
                            turn += 1
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)
        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
