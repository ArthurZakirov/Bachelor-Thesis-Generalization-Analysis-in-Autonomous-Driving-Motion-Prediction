import json
import os
import sys
sys.path.append('../converter_functions')
sys.path.append('../converter_maps')
from tqdm import tqdm
import pandas as pd
import numpy as np
from preprocessing_utils import ensure_dir, resize_map_mask
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.utils.splits import create_splits_scenes
from sklearn.model_selection import train_test_split
from preprocessing_utils import interpolate, remove_short_appearing_nodes
from data2Kitti_functions import append_crop_maps2df_samples, data2df_samples, data_velocity_stats, prepare_data_before_sample_creation
from nuScenes2data_functions import ns_scene2data, ns_map2data_map, nuScenes_category2Kitti_category, split_scene_names
from data2env_functions import shift_data
sys.path.append("../../Trajectron/trajectron")
from environment import Environment
scene_blacklist = [499, 515, 517]


def main(traj_config, map_config, data_path, output_dir, road_config):
    version = os.path.basename(data_path)
    dataroot = os.path.dirname(data_path)
    nusc = NuScenes(dataroot=dataroot, version=version)
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=None)

    val_split = 0.1
    ns_scene_names = split_scene_names(version, val_split)

    for split in ['train', 'val', 'test']:
        print(f'\nProcess {len(ns_scene_names[split])} {split}-scenes.')
        data_class_locs = {loc: list() for loc in road_config['locations']}
        for loc in road_config['locations']:
            scenes_dict = {speed: list() for speed in ['slow', 'middle', 'fast']}
            for ns_scene_name in ns_scene_names[split]:
                ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0])
                scene_loc = nusc.get('log', ns_scene['log_token'])['location']
                if scene_loc == loc:
                    data_class_locs[loc].append(ns_scene)
            for ns_scene in tqdm(data_class_locs[loc], desc= f'{split} {loc} scene', leave=False):
                scene_id = int(ns_scene['name'].replace('scene-', ''))
                if scene_id in scene_blacklist:  # Some scenes have bad localization
                    continue
                data = ns_scene2data(nusc, ns_scene, scene_id, env, nuScenes_category2Kitti_category)
                data, map_boundaries = shift_data(data, buffer=map_config['dim_clip'])
                data = prepare_data_before_sample_creation(data, traj_config['dt'], original_dt=0.5)
                scene_velocities = data_velocity_stats(data, dt=traj_config['dt'])
                df_samples = data2df_samples(data=data,
                                             ht=traj_config['history_time'],
                                             pt=traj_config['future_time'],
                                             dt=traj_config['dt'])
                if df_samples.empty:
                    continue
                nusc_map = NuScenesMap(dataroot=dataroot,
                                       map_name=nusc.get('log', ns_scene['log_token'])['location'])
                map_mask = ns_map2data_map(nusc_map,
                                           map_boundaries,
                                           map_config['fraction']).squeeze()
                df_samples = append_crop_maps2df_samples(df_samples,
                                            map_mask,
                                            map_config['dim_clip'],
                                            map_config['fraction'])
                if scene_velocities['max'] < road_config['speed_splits'][0]:
                    scenes_dict['slow'].append(df_samples)
                elif scene_velocities['max'] > road_config['speed_splits'][-1]:
                    scenes_dict['fast'].append(df_samples)
                else:
                    scenes_dict['middle'].append(df_samples)

            print(f'save {split} scenes in {loc}')
            for speed in ['slow', 'middle', 'fast']:
                if len(scenes_dict[speed]) >= 1:
                    df_samples_all_scenes = pd.concat(scenes_dict[speed], axis=0).reset_index(drop=True).to_dict()
                    file = f'nuScenes_{loc}_{speed}_{split}.json'
                    output_path = os.path.join(output_dir, file)
                    ensure_dir(output_dir)
                    with open(output_path, 'w') as f:
                        json.dump({'title': 'nuScenes', 'data': df_samples_all_scenes}, f)


if __name__ == '__main__':

    data_path = '../../datasets/raw/v1.0-mini/v1.0-mini'
    output_dir = '../../datasets/processed/Mantra_format/nuScenes_road_classes'
    ensure_dir(output_dir)
    traj_config = {
        'history_time': 2,
        'future_time': 6,
        'dt': 0.5
    }
    map_config = {
        'dim_clip': 90,
        'fraction': 1 / 2
    }
    road_config = {
        'locations': ['singapore-onenorth', 'boston-seaport', 'singapore-queenstown', 'singapore-hollandvillage'],
        'speed_splits': [30, 70]
    }

    main(traj_config, map_config, data_path, output_dir, road_config)
    print('finished nuScenes2Kitti')