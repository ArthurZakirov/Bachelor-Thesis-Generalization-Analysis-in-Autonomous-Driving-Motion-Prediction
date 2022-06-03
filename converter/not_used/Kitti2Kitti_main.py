import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
sys.path.append('../converter_functions')
from preprocessing_utils import ensure_dir
from preprocessing_utils import round_to_fraction

def interpolate_one_raw_track(node_data, dt):
    xy = np.array(node_data['trajectory'])
    if len(xy.shape) < 2:
        return None
    frame_id = np.arange(node_data['start'], node_data['end'] + 1)
    t_original = frame_id * 0.1

    t_start = round_to_fraction(t_original[0], dt)
    t_end = round_to_fraction(t_original[-1], dt)
    t_interp = np.arange(t_start, t_end + dt, dt)

    x_original = xy[0]
    x_interp = np.interp(t_interp, t_original, x_original)
    y_original = xy[1]
    y_interp = np.interp(t_interp, t_original, y_original)
    xy_interp = np.stack([x_interp, y_interp], axis=0)
    node_data['trajectory'] = xy_interp.tolist()
    return node_data

def interp_all_raw_tracks(tracks):
    tracks_interp = defaultdict(dict)
    for video_name, video_dict in tracks.items():
        for node_id, node_data in video_dict.items():
            node_data = interpolate_one_raw_track(node_data, dt=traj_config['dt'])
            if node_data is None:
                continue
            tracks_interp[video_name][node_id] = node_data
    return tracks_interp

def sort_tracks_by_speed(tracks, speed_dict):
    tracks_sorted = defaultdict(dict)
    for video_name, video_dict in tracks.items():
        video_id = int(video_name[35:39])
        for speed, speed_video_ids in speed_dict.items():
            if video_id in speed_video_ids:
                tracks_sorted[speed][video_name] = video_dict
    return tracks_sorted

def main():
    ###########################################
    ### PROCESS KITTI FOR MANTRA ##############
    ###########################################
    with open(os.path.join(data_dir, raw_file), 'r') as f:
        tracks = json.load(f)
    tracks = interp_all_raw_tracks(tracks)
    tracks = sort_tracks_by_speed(tracks, speed_dict)
    for speed in speed_dict.keys():
        title = f'Kitti_{speed}_interp'
        output_file = title + '.json'
        with open(os.path.join(output_dir_mantra, output_file), 'w') as f:
            json.dump({'title': title, 'data': tracks[speed]}, f)

    ###########################################
    ### PROCESS KITTI FOR TRAJECTRON ##########
    ###########################################
    with open(os.path.join(data_dir, raw_file), 'r') as f:
        tracks = json.load(f)
    tracks = sort_tracks_by_speed(tracks, speed_dict)
    for speed in speed_dict.keys():
        title = f'Kitti_{speed}'
        output_file = title + '.json'
        with open(os.path.join(output_dir_trajectron, output_file), 'w') as f:
            json.dump({'title': title, 'data': tracks[speed]}, f)

if __name__ == '__main__':
    data_dir = '../../datasets/raw/Kitti/trajectories'
    raw_file = 'kitti_dataset.json'
    output_dir_mantra = '../../datasets/processed/Mantra_format/Kitti_road_classes/trajectories'
    output_dir_trajectron = '../../datasets/raw/Kitti/trajectories'
    ensure_dir(output_dir_mantra)
    ensure_dir(output_dir_trajectron)
    speed_dict = {
        'middle': [1, 2, 5, 9, 11, 13, 14, 17, 18, 48, 51, 56, 57, 59, 60, 84, 91, 93],
        'fast': [15, 27, 28, 29, 32, 52, 70]
    }
    traj_config = {
        'dt': 0.5
    }
    main()
    print('finished Kitti2Kitti')