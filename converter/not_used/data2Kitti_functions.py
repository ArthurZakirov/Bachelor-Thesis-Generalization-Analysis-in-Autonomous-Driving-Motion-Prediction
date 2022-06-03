import json
import sys
import os
from tqdm import tqdm

import torch
from pyquaternion import Quaternion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.append('../Trajectron-plus-plus/trajectron')
from preprocessing_utils import normalize_trajectory, remove_short_appearing_nodes, interpolate
import cv2


def prepare_data_before_sample_creation(data, dt, original_dt):
    data['t'] = data['frame_id'].astype(float) * original_dt
    data['scene_id'] = str(10000 + int(data.loc[0, 'scene_id']))[1:]

    data = remove_short_appearing_nodes(data,
                                        min_node_time=1.,
                                        id_feature='node_id',
                                        t_feature='t')
    data = interpolate(data,
                       dt=dt,
                       interp_features=["x", "y"],
                       categoric_features=["type", "scene_id"],
                       id_feature='node_id',
                       t_feature='t')
    return data


def data2df_samples(data, ht, pt, dt):
    """
    Arguments
    ---------
    data : pd.DataFrame
        format similiar to nuScenes "data" in T++ pipeline (timestepwise numerical_features and categorical_features)

    ht : float
        history time, measured in seconds
    pt : float
        prediction time, measured in seconds
    dt : flat
        sampling time, measured in seconds

    Returns
    -------
    df_samples: pd.DataFrame
        timestepwise samples of (hist, fut, relative position in map, relative angle in map, categorical_features)
    """
    pl = int(pt/dt)
    hl = int(ht/dt)
    df_samples = pd.DataFrame()
    sample_id = 0
    num_vec = 0
    for ns_node_id in tqdm(list(data['node_id'].unique()), desc='process node', leave=False):
        node_df = data[data['node_id'] == ns_node_id].copy()
        node_df.reset_index(drop=True, inplace=True)
        xy_test = node_df[['x', 'y']].values

        # remove not moving nodes
        if np.linalg.norm(xy_test[0]-xy_test[-1]) < 1:
            continue
        # remove nodes with less than 6 sec observation
        if not (len(node_df) >= hl+pl+1):
            continue
        # remove pedestrians
        if node_df.loc[0, 'type'] in ['Pedestrian', 'PEDESTRIAN', 'Cyclist', 'Tram']:
            continue
        elif node_df.loc[0, 'type'] == 'VEHICLE':
            node_df.loc[:, 'type'] = 'Car'

        for ts in range(hl, len(node_df)-pl):
            xy = node_df[['x', 'y']].values
            xy, course, pos = normalize_trajectory(xy, ts)
            hist = xy[ts-hl : ts]
            fut  = xy[ts    : ts+pl]
            
            sample = {'index': ts,
                      'pasts': [hist.tolist()], 
                      'futures': [fut.tolist()],
                      'presents': [pos.tolist()],
                      'angle_presents': -course.item(),
                      'video_track': node_df.loc[0, 'scene_id'],
                      'vehicles': node_df.loc[0, 'type'],
                      'number_vec': num_vec
                     }
            df_sample = pd.DataFrame(index=[sample_id], data=sample)
            df_samples = df_samples.append(df_sample)
            
            sample_id += 1
        num_vec += 1
    return df_samples


def append_crop_maps2df_samples(df_samples, map_mask, dim_clip, fraction):
    """
    Arguments
    ---------
    df_samples : pd.DataFrame
        timestepwise samples of (hist, fut, relative position in map, relative angle in map, categorical_features)

    map_mask : np.array[y_size, x_size]
        one layer map mask with y_size being first

    dim_clip : float
        size of cropped map around vehicle in each direction, measured in meters

    fraction : float
        step size of map

    Returns
    -------
    df_samples_with_maps : pd.DataFrame
        timestepwise samples of (hist, fut, relative position in map, relative angle in map, categorical_features, map, rotated_map)
    """
    cropped_maps = list()
    cropped_rotated_maps_OH = list()
    dim_clip = int(dim_clip / fraction)
    for i in tqdm(range(len(df_samples)), desc='apend map', leave=False):
        origin = df_samples.loc[i, 'presents']
        angle = df_samples.loc[i, 'angle_presents']
        x_current = int(origin[0] / fraction)
        y_current = int(origin[1] / fraction)
        # print("map_mask.shape:", map_mask.shape)
        # print("y,x:", y_current, x_current)
        scene_track_clip = map_mask[
            y_current - dim_clip : y_current + dim_clip,
            x_current - dim_clip : x_current + dim_clip]
        matRot_scene = cv2.getRotationMatrix2D(center=(dim_clip, dim_clip), angle=-angle, scale=1)
        scene_track_onehot_clip = cv2.warpAffine(
            scene_track_clip,
            matRot_scene,
            (2*dim_clip, 2*dim_clip),
            borderValue=0,
            flags=cv2.INTER_NEAREST)

        cropped_maps.append(scene_track_clip.astype(int))#.tolist())
        cropped_rotated_maps_OH.append(scene_track_onehot_clip.astype(int))#.tolist())

    df_samples_with_maps = df_samples.copy()
    df_samples_with_maps['scene'] = cropped_maps
    df_samples_with_maps['scene_crop'] = cropped_rotated_maps_OH
    return df_samples_with_maps



def data_map2Kitti_map(map_mask):
    """ categorical OHE [0,0,1,0...,0] -> numerical OHE [1,2,3,4,...]
    Arguments
    ---------
    map_mask : np.array [layers, y_size, x_size]
        OHE-layers for different area types

    Returns
    -------
    map_mask : np.array [y_size, x_size]
    """

    # Deal with overlapping: remove field from all categories
    overlapping_fields = (map_mask.sum(axis=0, keepdims=True) > 1).astype(int)
    map_mask = (map_mask - overlapping_fields).clip(min=0)

    # categorical OHE [0,0,1,0...,0] -> numerical OHE [1,2,3,4,...]
    num_layers = len(map_mask)
    numeric_OH = np.arange(1, num_layers + 1).reshape(num_layers, 1, 1)
    map_mask = (map_mask * numeric_OH).sum(0)
    return map_mask


def data_velocity_stats(data, dt, k=1):
    node_ids = data['node_id'].unique()
    node_groups = data.groupby('node_id')
    velocities = list()

    for node_id in node_ids:
        node_df = node_groups.get_group(node_id)
        xy = node_df[['x','y']]
        pos = np.linalg.norm(xy.values, axis=1)
        v_max = np.abs(np.diff(pos)).max() / dt * 3.6
        if str(node_df['type']) == 'PEDESTRIAN':
            continue
        if v_max is None:
            continue
        velocities.append(v_max)
    velocities = np.sort(np.array(velocities))
    velocity_stats = {
        'mean' : velocities.mean(),
        'max': velocities[-k:].mean(),
        'min': velocities[:k].mean(),
    }
    return velocity_stats

