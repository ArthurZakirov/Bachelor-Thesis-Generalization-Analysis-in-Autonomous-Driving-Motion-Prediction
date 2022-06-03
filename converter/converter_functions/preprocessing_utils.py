"""
Module that contains utils used in several converter pipelines
"""
import os
import numpy as np
import pandas as pd
import cv2


def round_to_fraction(x, fraction):
    """
    round float number to a float number of a certain fraction
    """
    return np.round(x / fraction) * fraction


def rot(xy, alpha):
    """Rotiere die Punktwolke um alpha(deg) gegen den Uhrzeigersinn

    Arguments
    ---------
    xy : np.array[N, 2]
    alpha : np.array[]
        deg °

    Returns
    -------
    xy_rot : np.array[N,2]
    """
    alpha = alpha * np.pi / 180
    T = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    xy_rot = xy @ T.T
    return xy_rot


def angle_between(v_0, e):
    """Angle against the clock from v_0 to e in deg °

    Arguments
    ---------
    v_0 : np.array[2]
    e : np.array[2]

    Returns
    -------
    alpha : np.array[]
    """
    alpha = (
        np.arctan2(v_0[0] * e[1] - v_0[1] * e[0], v_0[0] * e[0] + v_0[1] * e[1])
        * 180
        / np.pi
    )
    return alpha


def normalize_trajectory(xy, ts):
    """
    Arguments
    ---------
    xy : np.array [ht + pt, 2]
        array containing the x and y coordinate in a global coordinate system
    ts : int
        timestep, index inside array, not in seconds

    Returns
    -------
    xy_norm : np.array [ht + pt, 2]
    course : np.array[]
    pos : np.array[]
    """
    pos = xy[ts]
    xy = xy - pos
    v_ts = xy[ts + 2] - xy[ts]
    north = np.array([0, 1])
    course = angle_between(v_ts, north)
    xy_norm = rot(xy, course)
    return xy_norm, course, pos


def remove_short_appearing_nodes(df, min_node_time, id_feature, t_feature):
    """
    Arguments
    ---------
    df : pd.DataFrame
        "raw" version of the "data" format from nuScenes T++ pipeline
        difference: df yet includes nodes with too short appearing times, which makes interpolation impossible

    min_node_time : float
        minimum time a node appears for in a scene, measured in seconds
        must be larger than "dt" because otherwise largest and smallest value will be rounded to the same value

    Returns
    -------
    df_without_short : pd.DataFrame
    """
    node_appearence_time = df.groupby(id_feature)[t_feature].apply(
        lambda df_node: df_node.max() - df_node.min()
    )
    short_appearing_nodes = node_appearence_time[
        node_appearence_time < min_node_time
    ].index.values
    mask = df[id_feature].isin(short_appearing_nodes)
    df_without_short = df.copy()
    df_without_short.drop(index=df_without_short[mask].index, inplace=True)
    return df_without_short


def interpolate(df, dt, interp_features, categoric_features, id_feature, t_feature):
    """Interpolate features listed in interp_features to the sample time: dt

    Arguments
    ---------
    df : pd.DataFrame
        "raw" version of the "data" format from nuScenes T++ pipeline
        difference: samples are not yet on the required sample time

    interp_features : ['feature_1', feature_2',...]
        numerical features names that should be interpolated to the new dt
        the time and frame_id must NOT be listed here!

    categoric_features : ['feature_1', feature_2',...]
        features names that should not be interpolated, instead the should be copied to the new samples
        the node id must NOT be listed here!

    id_feature : str
        feature name that describes the identity of a node

    t_feature : str
        feature name that describes the time of the df at the previous dt

    dt : float
        new required sample time

    Returns
    -------
    df_interp : pd.DataFrame
    """
    df_interp = pd.DataFrame()
    objIDs = df[id_feature].unique()

    for objID in objIDs:
        df_obj = df[df[id_feature] == objID].copy()
        df_obj.drop(columns=id_feature, inplace=True)
        df_obj.reset_index(drop=True, inplace=True)

        t_original = df_obj[t_feature].values
        t_start = round_to_fraction(t_original[0], dt)
        t_end = round_to_fraction(t_original[-1], dt)
        t_interp = np.arange(t_start, t_end + dt, dt)

        if t_start == 0:
            frame_ids = (t_interp[1:] / dt).astype(int)
            frame_ids = np.insert(frame_ids, 0, 0)
        else:
            frame_ids = (t_interp / dt).astype(int)

        df_obj_new = pd.DataFrame()
        df_obj_new[t_feature] = t_interp
        df_obj_new[id_feature] = objID
        df_obj_new["frame_id"] = frame_ids

        for feature in interp_features:
            feature_original = df_obj.loc[:, feature].values
            feature_interp = np.interp(t_interp, t_original, feature_original)
            df_obj_new.loc[:, feature] = feature_interp

        for feature in categoric_features:
            df_obj_new.loc[:, feature] = df_obj.loc[0, feature]

        df_interp = pd.concat([df_interp, df_obj_new])

    df_interp.sort_values(t_feature, ascending=True, inplace=True)
    df_interp.reset_index(drop=True, inplace=True)
    return df_interp


def remove_not_moving_nodes_from_scene(
    scene, crit_norm_vehicle=2, crit_norm_pedestrian=0
):
    """Remove nodes that move less than a certain threshold, besides robot node

    Arguments
    ---------
    scene : pd.DataFrame
    crit_norm_vehicle : int
        Critical travelled distance of vehicle before it counts as 'not moving'

    crit_norm_pedestrian : int
        Critical travelled distance of pedestrian before it counts as 'not moving'

    Returns
    -------
    scene : pd.DataFrame
    """
    not_moving_nodes_list = list()
    node_ids = [node_id for node_id in scene.node_id.unique() if node_id != "ego"]

    for node_id in node_ids:
        node_df = scene[scene.node_id == node_id]

        # calculate Total Movement
        x_dist = node_df.x.max() - node_df.x.min()
        y_dist = node_df.y.max() - node_df.y.min()
        dist_vec = np.stack([x_dist, y_dist])
        dist_norm = np.linalg.norm(dist_vec)

        if node_df.iloc[0].type == "VEHICLE":
            node_moving = dist_norm > crit_norm_vehicle

        elif node_df.iloc[0].type == "PEDESTRIAN":
            node_moving = dist_norm > crit_norm_pedestrian

        # eliminate not moving nodes
        if not node_moving:
            not_moving_nodes_list.append(node_id)

    if len(not_moving_nodes_list) != 0:
        select_not_moving_nodes = scene[scene.node_id.isin(not_moving_nodes_list)].index
        scene = scene.drop(select_not_moving_nodes)

    scene = scene.reset_index(drop=True)
    return scene


def resize_map_mask(map_mask, factor):
    """
    Arguments
    ---------
    map_mask : np.array[y_size, x_size]
    factor : flaot

    Returns
    -------
    map_mask : np.array[y_size*factor, x_size*factor]
    """
    dsize = tuple((np.array(map_mask.shape) * factor).astype(int))
    dsize = dsize[1], dsize[0]
    map_mask = cv2.resize(map_mask, dsize=dsize)
    return map_mask


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
