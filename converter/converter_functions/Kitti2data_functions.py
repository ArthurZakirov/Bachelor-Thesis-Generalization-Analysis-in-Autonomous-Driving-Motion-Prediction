"""
Module that contains functions for the processing from Kitti raw to "data" format
"""
import sys
import os
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from preprocessing_utils import (
    interpolate,
    angle_between,
    remove_short_appearing_nodes,
    remove_not_moving_nodes_from_scene,
)
from preprocessing_utils import resize_map_mask

sys.path.append("../converter_maps")
sys.path.append("../../Trajectron/trajectron")
from DataMap import DataMap
from environment import derivative_of

NATIVE_FRACTION = 0.5
NATIVE_DT = 0.1


def sort_tracks_by_speed(tracks, speed_dict):
    tracks_sorted = defaultdict(dict)
    for video_name, video_dict in tracks.items():
        video_id = int(video_name[35:39])
        for speed, speed_video_ids in speed_dict.items():
            if video_id in speed_video_ids:
                tracks_sorted[speed][video_name] = video_dict
    return tracks_sorted


def kitti_map2data_map(map_mask):
    # m = np.eye(5)[map_mask].transpose(2, 0, 1)
    # map_mask = np.stack([m[1], m[1], m[2]], axis=0)
    map_mask = np.expand_dims((map_mask == 1), axis=0).astype(int)
    return map_mask


def Kitti2data(args, speed):
    # MAP PREPROCESSING
    map_dir = os.path.join(args.raw_data_dir, "maps")
    map_names = os.listdir(map_dir)
    map_names.pop(-1)
    print(f"Preprocess {len(map_names)} maps.")
    map_dict = dict()
    for map_name in tqdm(map_names):
        map_path = os.path.join(map_dir, map_name)
        map_mask = cv2.imread(map_path, 0)
        map_mask = resize_map_mask(map_mask, factor=NATIVE_FRACTION / args.fraction)
        map_mask = np.expand_dims((map_mask == 1), axis=0).astype(int)

        data_map = DataMap(map_mask, args)
        map_dict[map_name] = data_map

    # TRAJECTORIES PREPROCESSING
    scene_id_dict = {
        "middle": [1, 2, 5, 9, 11, 13, 14, 17, 18, 48, 51, 56, 57, 59, 60, 84, 91, 93],
        "fast": [15, 27, 28, 29, 32, 52, 70],
    }

    traj_path = os.path.join(args.raw_data_dir, "trajectories", "kitti_dataset.json")
    with open(traj_path, "r") as f:
        all_tracks = json.load(f)
    tracks_by_speed = sort_tracks_by_speed(all_tracks, scene_id_dict)

    traj_dict = defaultdict(list)
    for split in ["train", "test", "val"]:
        tracks = tracks_by_speed[speed]
        print(f"Preprocess {len(tracks)} {split} scenes.")
        for i, video_name in enumerate(tqdm(tracks)):
            # Trajectories
            df_video = pd.DataFrame(tracks[video_name]).transpose()
            df = []

            if df_video.loc["track_0", "end"] * NATIVE_DT < args.scene_time:
                continue
            for node_id, node_data in df_video.iterrows():
                category = node_data["cls"]
                if category in ["Car", "Truck", "Van"]:
                    our_category = "VEHICLE"
                elif category == "Pedestrian":
                    our_category = "PEDESTRIAN"
                else:
                    continue
                xy = np.array(node_data["trajectory"]).T
                if xy.ndim == 1:
                    continue
                v = derivative_of(xy, NATIVE_DT)
                course = [
                    angle_between(v[ts], e=np.array([0, 1])) / 360 * 2 * np.pi
                    for ts in range(len(v))
                ]

                frame_id = np.arange(node_data["start"], node_data["end"] + 1)
                node_df = pd.DataFrame(
                    {
                        "scene_id": i,
                        "t": frame_id * NATIVE_DT,
                        "node_id": "ego" if node_id == "track_0" else node_id,
                        "x": xy[:, 0],
                        "y": xy[:, 1],
                        "heading": course,
                        "type": our_category,
                        "robot": True if node_id == "track_0" else False,
                    }
                )
                df.append(node_df)
            data = pd.concat(df, axis=0).reset_index(drop=True)
            data = remove_short_appearing_nodes(
                data,
                min_node_time=args.min_node_time,
                id_feature="node_id",
                t_feature="t",
            )
            data = remove_not_moving_nodes_from_scene(data)
            data = interpolate(
                data,
                dt=args.dt,
                interp_features=["x", "y", "heading"],
                categoric_features=["scene_id", "type", "robot"],
                id_feature="node_id",
                t_feature="t",
            )
            traj_take = [data]

            date = video_name[6:16]
            video_id = video_name[-9:-5]
            map_name = f"{date}__{date}_drive_{video_id}_sync_map.png"
            traj_dict[split].append((traj_take, map_name))
    return traj_dict, map_dict
