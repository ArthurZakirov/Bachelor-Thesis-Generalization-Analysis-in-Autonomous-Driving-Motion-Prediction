import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import sys
sys.path.append('../converter_functions')
sys.path.append('../converter_maps')
import json
from preprocessing_utils import ensure_dir
from openDD2data_functions import openDD2data
from data2env_functions import shift_data
from preprocessing_utils import interpolate, remove_short_appearing_nodes
from data2Kitti_functions import data2df_samples, data_map2Kitti_map, append_crop_maps2df_samples, prepare_data_before_sample_creation


def openDD_dict2Kitti(traj_dict, map_dict, traj_config, map_config, output_dir):
    for split in ['train', 'val', 'test']:
        df_samples_all_scenes = list()
        for take, map_name in tqdm(traj_dict[split][:1], desc=f'{split} take'):
            num_scenes = len([scene for take, _ in traj_dict[split] for scene in take])
            data_map = map_dict[map_name]
            for data in tqdm(take, desc=f'{split} scene'):
                #sys.stdout.write("\rProcessed {}/{} {} scenes.".format(len(df_samples_all_scenes), num_scenes, split))
                data, scene_boundaries = shift_data(data=data, buffer=data_map.scene_buffer)
                data = prepare_data_before_sample_creation(data, traj_config['dt'])

                df_samples = data2df_samples(data=data,
                                             ht=traj_config['history_time'],
                                             pt=traj_config['future_time'],
                                             dt=traj_config['dt'])
                if df_samples.empty:
                    continue

                scene_mask = data_map.extract_scene_mask_from_map_mask(scene_boundaries)# [0=lane, 1=area, 2=border]
                scene_mask = np.max(scene_mask[:2], axis=0).squeeze()

                df_samples = append_crop_maps2df_samples(df_samples=df_samples,
                                                         map_mask=scene_mask,
                                                         dim_clip=map_config['dim_clip'],
                                                         fraction=map_config['fraction'])

                df_samples_all_scenes.append(df_samples)
                #sys.stdout.write("\rProcessed {}/{} {} scenes.".format(len(df_samples_all_scenes), num_scenes, split))
        print(f'save {split} data..')
        df_samples_all_scenes = pd.concat(df_samples_all_scenes, axis=0).reset_index(drop=True).to_dict()
        file = f'openDD_{split}.json'
        output_path = os.path.join(output_dir, file)
        with open(output_path, 'w') as f:
            json.dump({'title': 'openDD', 'data': df_samples_all_scenes}, f)
    print('openDD2Kitti finished!')


def main():
    print('OPENDD -> DATA:')
    traj_dict, map_dict = openDD2data(data_dir,
                                      traj_config_openDD2data,
                                      map_config_openDD2data)
    print('\n\n\n\n\n\nDATA -> KITTI')
    openDD_dict2Kitti(traj_dict,
                      map_dict,
                      traj_config_data2kitti,
                      map_config_data2kitti,
                      output_dir)

if __name__ == '__main__':
    # The pipeline from openDD to Kitti consists of 2 seperate steps
    # 1) openDD -> "data"   (like in the T++ pipeline for nuScenes)
    # 2) data -> df_samples (kitti format)
    #
    # The first set of "traj_config, map_config" are for the first conversion step
    # The second set of "traj_config, map_config" are for the second conversion step

    data_dir = '../../datasets/raw/openDD'
    output_dir = '../../datasets/processed/Mantra_format/openDD_full'
    ensure_dir(output_dir)

    traj_config_openDD2data = {
        'scene_time': 20,       # (sec) duration of scene
        'node_types_str': ["Car", "Pedestrian", "Medium Vehicle", "Trailer", "Bus", "Heavy Vehicle"],
        'dt': 0.5,               # (sec) Convert to nuScenes sample frequency first and then to kitti frequency from there
        'max_scene_overlap': 2,  # (sec) How many seconds can the scenes overlap. Sometimes that is necessary, because we don't want to miss scenes due to limited robot nodes.
        'crit_norm_vehicle': 1,  # (meter) Minimum travelled distance of a vehicle to be not considered "parked"
    }

    map_config_openDD2data = {
        "fraction": 1 / 2,      # (meter), step size of map
        "lane_radius": 4,       # (meter), width of lane
        "map_buffer": 100,      # padding around original map. Should be higher than scene_buffer.
        "scene_buffer": 90      # (meter), extra space added to the maximum positions of vehicles in a scene.
    }
    traj_config_data2kitti = {
        'history_time': 2,      # (sec), time length of observed trajectory
        'future_time': 6,       # (sec), time length of predicted trajectory
        'dt': 0.5               # (sec), timestep size
    }
    map_config_data2kitti = {
        'dim_clip': 90,   # (meter), size of cropped map around vehicle in each direction -> [2*dim_clip/fraction, 2*dim_clip/fraction]
        'fraction': map_config_openDD2data['fraction']
    }
    main()
    print('finished openDD2Kitti')