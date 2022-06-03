import json
import os
import sys
import random
sys.path.append('../converter_functions')
sys.path.append('../converter_maps')
from tqdm import tqdm
import pandas as pd
import numpy as np
from preprocessing_utils import ensure_dir
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from lyft_dataset_sdk.lyftdataset import LyftDataset

from nuscenes.utils.splits import create_splits_scenes
from sklearn.model_selection import train_test_split
from preprocessing_utils import interpolate, remove_short_appearing_nodes
from data2Kitti_functions import append_crop_maps2df_samples, data2df_samples, data_velocity_stats, prepare_data_before_sample_creation
from nuScenes2data_functions import ns_scene2data, ns_map2data_map, nuScenes_category2Kitti_category
from data2env_functions import shift_data, init_env
from lyft2data_functions import process_lyft_map, lyft_category2Kitti_category
from DataMap import DataMap
sys.path.append("../../Trajectron/trajectron")
from environment import Environment

def main():
    print('Start Map processing.')
    map_mask = process_lyft_map(map_path, map_config_lyft2data)
    data_map = DataMap(map_mask, map_config_lyft2data)
    nusc = LyftDataset(data_path=traj_path, json_path=os.path.join(traj_path, f'train_data'))
    num_scenes = len(nusc.scene)
    scene_ids = list(range(num_scenes))
    random.shuffle(scene_ids)
    splits_dict = {
        'train': [nusc.scene[i] for i in scene_ids[:int(num_scenes * (1 - val_split))]],
        'val': [nusc.scene[i] for i in scene_ids[-int(num_scenes * val_split):]]
    }
    for split, split_scenes in splits_dict.items():
        print(f'Process {len(split_scenes)} {split}-scenes')
        scenes_dict = {speed: list() for speed in ['slow', 'middle', 'fast']}
        for scene_id, ns_scene in enumerate(tqdm(split_scenes, leave=False, desc=f'{split} scene')):
            #sys.stdout.write("\rcreate dataframe...")
            data = ns_scene2data(nusc, ns_scene, str(scene_id), init_env(), lyft_category2Kitti_category)
            data, scene_boundaries = shift_data(data, buffer=data_map.scene_buffer)
            data = prepare_data_before_sample_creation(data, dt=traj_config['dt'], original_dt=0.2)

            scene_velocities = data_velocity_stats(data, dt=traj_config['dt'])
            #sys.stdout.write("\rcreate samples...  ")
            df_samples = data2df_samples(data=data,
                                         ht=traj_config['history_time'],
                                         pt=traj_config['future_time'],
                                         dt=traj_config['dt'])
            #sys.stdout.write("\rcreate map...      ")
            scene_mask = data_map.extract_scene_mask_from_map_mask(scene_boundaries)
            scene_mask = scene_mask.squeeze()
            # df_samples = append_crop_maps2df_samples(df_samples,
            #                                          scene_mask,
            #                                          map_config_data2Kitti['dim_clip'],
            #                                          map_config_data2Kitti['fraction'])
            # if df_samples.empty:
            #     continue
            if scene_velocities['max'] < road_config['speed_splits'][0]:
                scenes_dict['slow'].append((df_samples, scene_mask))
            elif scene_velocities['max'] > road_config['speed_splits'][-1]:
                scenes_dict['fast'].append((df_samples, scene_mask))
            else:
                scenes_dict['middle'].append((df_samples, scene_mask))

        print(f'Save {split} scenes..')
        for speed in ['slow', 'middle', 'fast']:
            df_samples_all_scenes, scene_mask = scenes_dict[speed]
            if len(scenes_dict[speed]) >= 1:
                dict_samples_all_scenes = pd.concat(df_samples_all_scenes, axis=0).reset_index(drop=True).to_dict()
                scene_mask_list = scene_mask.tolist()
                file = f'lyft_{speed}_{split}.json'
                output_path = os.path.join(output_dir, file)
                with open(output_path, 'w') as f:
                    json.dump({'title': 'lyft', 'data': dict_samples_all_scenes, 'map': scene_mask_list}, f)

if __name__ == '__main__':
    val_split = 0.1
    output_dir = '../../datasets/processed/Mantra_format/lyft_road_classes'
    traj_path = '../../datasets/raw/lyft'
    map_path = '../../datasets/raw/lyft/maps/map_raster_palo_alto.png'
    ensure_dir(output_dir)

    traj_config = {
        'history_time': 2,
        'future_time': 6,
        'dt': 0.5
    }
    road_config = {
        'speed_splits': [30, 65]
    }
    map_config_lyft2data = {
        'fraction': 1 / 2,
        'map_buffer': 100,
        'scene_buffer': 90,
        'area_colours': {
            'vehicle_road': [128, 128, 128],
            'pedestrian_zebra': [250, 235, 215],
            'pedestrian_green': [0, 201, 131],
            # 'pedestrian_only':[211, 211, 211]
        }
    }
    map_config_data2Kitti = {
        'dim_clip': 90, # (meter), size of cropped map around vehicle in each direction -> [2*dim_clip/fraction, 2*dim_clip/fraction]
        'fraction': map_config_lyft2data['fraction']
    }
    main()
    print('finished lyft2Kitti')