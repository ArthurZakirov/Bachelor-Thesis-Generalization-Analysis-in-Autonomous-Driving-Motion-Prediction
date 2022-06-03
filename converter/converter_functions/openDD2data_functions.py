"""
Module that contains functions for the processing from openDD raw to "data" format
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import argrelextrema
import sqlite3
from matplotlib.path import Path
from preprocessing_utils import interpolate, remove_not_moving_nodes_from_scene

sys.path.append("../converter_maps")
from DataMap import DataMap


#######################################################################################
###   EXECUTE   #######################################################################
#######################################################################################
def openDD2data(args):
    ##################################
    # MAP DICT
    ##################################
    map_names = os.listdir(args.raw_data_dir)

    map_dict = dict()
    print(f"Process {len(map_names)} maps.")
    for map_name in tqdm(map_names, position=0, leave=True):
        map_path = os.path.join(
            *[
                args.raw_data_dir,
                map_name,
                "map_" + map_name,
                "map_" + map_name + ".sqlite",
            ]
        )
        data_map = preprocess_openDD_map(map_path, map_name, args)
        map_dict[map_name] = data_map

    ##################################
    # TRAJECTORY DICT
    ##################################
    split_paths = {
        "train": "r_1_train.txt",
        "val": "r_1_val.txt",
        "test": "r_A.txt",
    }
    traj_dict = {}
    for split, file in split_paths.items():

        # get list of scene names for each split
        split_definition = (
            "rdb1/my_split_definition_mini" if args.mini else "rdb1/my_split_definition"
        )
        with open(os.path.join(args.raw_data_dir, split_definition, file), "r") as f:
            takes_list = f.readlines()
        takes_list = [take_name.strip() for take_name in takes_list]
        print(f"Process {len(takes_list)} {split} takes.")

        traj_dict[split] = []
        for take_name in tqdm(takes_list, position=0, leave=True):
            map_name = take_name.split("_")[0]
            # load trajectories df
            traj_path = os.path.join(
                *[
                    args.raw_data_dir,
                    map_name,
                    "trajectories_" + map_name + "_v3.sqlite",
                ]
            )
            df = load_df(traj_path, take_name)
            df = prepare_df(df=df, map=map_dict[map_name], args=args)

            traj_take = create_scenes_from_df(
                df=df,
                scene_time=args.scene_time,
                dt=args.dt,
                max_scene_overlap=args.max_scene_overlap,
                crit_norm_vehicle=args.crit_norm_vehicle,
            )
            if len(traj_take) == 0:
                continue
            traj_dict[split].append((traj_take, map_name))
    return traj_dict, map_dict


def load_df(traj_path, take_name):
    """Load original openDD trajectories Dataframe

    Arguments
    ---------
    traj_path : str

    Returns
    -------
    df : pd.DataFrame
    If there are no potential robot nodes in the take, return None.
    """
    cnx = sqlite3.connect(traj_path)
    df = pd.read_sql_query("SELECT * FROM " + take_name, cnx)
    cnx.close()

    return df


########################################################################################
###    PREPARE   DF     ################################################################
########################################################################################
def prepare_df(df, map, args):
    df = change_features(df)
    df = extract_node_types(df, node_types_str=args.node_types_str)
    df = remove_short_appearing_nodes(df, min_node_time=args.min_node_time)
    df = remove_outside_of_map(df, map_boundaries=map.map_boundaries)
    df = interpolate(
        df,
        dt=args.dt,
        interp_features=["x", "y", "heading"],
        categoric_features=["type", "robot"],
        id_feature="node_id",
        t_feature="TIMESTAMP",
    )
    return df


def apply_convert_angle(angle):
    """Shift angle range from [0, 2*pi] to [-pi, pi]

    Arguments
    ---------
    angle :  np.array ([])
        [0, 2*pi]

    Returns
    -------
    angle : np.array ([])
        [-pi, pi]
    """
    if angle >= np.pi:
        new_angle = np.pi - angle
    else:
        new_angle = angle
    return new_angle


def apply_rename_node_types(node_type_str):
    """Renames node types from openDD convention to nuScenes Convention
    Arguments
    ---------
    node_type_str : str
        openDD naming convention ['Car', 'Pedestrian']

    Returns
    -------
    nuScenes_node_type_str : str
        nuScenes naming convtion ['VEHICLE', 'PEDESTRIAN']

    """
    if node_type_str == "Pedestrian":
        nuScenes_node_type_str = "PEDESTRIAN"
    else:
        nuScenes_node_type_str = "VEHICLE"
    return nuScenes_node_type_str


def change_features(df):
    """Drop unnecessary columns and append new ones that are required for later processing

    Arguments
    ---------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
    """
    # append new columns
    df.loc[:, "robot"] = False
    df.loc[:, "frame_id"] = 0
    df.loc[:, "scene_id"] = 0

    # rename some columns
    df.rename(
        columns={
            "UTM_X": "x",
            "UTM_Y": "y",
            "UTM_ANGLE": "heading",
            "CLASS": "type",
            "OBJID": "node_id",
        },
        inplace=True,
    )
    # drop some columns
    df.drop(
        columns=["V", "ACC", "ACC_LAT", "ACC_TAN", "WIDTH", "LENGTH", "TRAILER_ID"],
        inplace=True,
    )

    # convert angle format
    df.loc[:, "heading"] = df.loc[:, "heading"].apply(apply_convert_angle)
    return df


def extract_node_types(df, node_types_str):
    """Only Keep nodes in the df that ae listed in the node_types list.

    Arguments
    --------
    df : pd.DataFrame
    node_types_str : list of node_type_str
        openDD names of requested node_types
        ['Car', 'Pedestrian']
    """
    idx = False
    for node_type in node_types_str:
        idx = idx | (df.type == node_type)
    df = df.loc[idx].copy()
    df.loc[:, "type"] = df.loc[:, "type"].apply(apply_rename_node_types)
    return df


def remove_short_appearing_nodes(df, min_node_time=2.0):
    node_appearence_time = df.groupby("node_id")["TIMESTAMP"].apply(
        lambda df_node: df_node.max() - df_node.min()
    )
    short_appearing_nodes = node_appearence_time[
        node_appearence_time < min_node_time
    ].index.values
    mask = df["node_id"].isin(short_appearing_nodes)
    df_copy = df.copy()
    df_copy.drop(index=df_copy[mask].index, inplace=True)
    return df_copy


def remove_outside_of_map(df, map_boundaries):
    """Remove data points of nodes that are outside of the map
    Arguments
    ---------
    df : pd.DataFrame
    map : OpenDDMap

    Returns
    -------
    df : pd.DatatFrame
    """
    x_min, x_max, y_min, y_max = map_boundaries
    inside_the_map = (
        (df["x"] > x_min) & (df["x"] < x_max) & (df["y"] > y_min) & (df["y"] < y_max)
    )
    return df[inside_the_map]


def round_to_fraction(x, fraction):
    return np.round(x / fraction) * fraction


########################################################################################
###    CREATE SCENES    ################################################################
########################################################################################
def create_scenes_from_df(df, scene_time, dt, max_scene_overlap, crit_norm_vehicle):
    """
    Arguments
    ---------
    df : pd.DataFrame
        columns = ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading']

        Full preprocessed openDD Dataframe

    scene_time : int
        Duration of scene in seconds (not timestamps)

    Returns
    -------
    scenes : list of scene
        scene : pd.DataFrame
            columns : ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading', 'scene_id', 'frame_id']

            A scene starts with the first timestamp of a node that appears for at least scene_time seconds.
            It last for scene_time. The scene has a scene_id.
            Multiple Filters to that scene are applied such as:
            'remove_not_moving_nodes()', 'set_robot_node()', 'reset_node_id()', 'set_frame_id()'

    """
    df_robots = get_robot_first_appearences(df, scene_time, max_scene_overlap)
    if df_robots.empty:
        return []
    scenes = []
    for scene_id, robot_node in df_robots.iterrows():
        scene = extract_scene_from_df(
            df=df,
            timestamp=robot_node["TIMESTAMP"],
            scene_time=scene_time,
            dt=dt,
            scene_id=scene_id,
        )

        scene = set_robot_node(scene, node_id=robot_node["node_id"])

        scene = remove_not_moving_nodes_from_scene(scene, crit_norm_vehicle)

        scene = reset_node_id(scene)

        scene = set_frame_id(scene)

        scenes.append(scene)

    return scenes


def get_robot_first_appearences(df, scene_time, max_scene_overlap=2):
    """Get dict of robot nodes in a 1/scene_time frequency

    Arguments
    ---------
    df : pd.DataFrame
        columns = ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading']
        Full preprocessed openDD Dataframe


    scene_time : int
        Duration of scenes in seconds (not timesteps)

    max_scene_overlap : int
        Nodes that appear for at least scene_time seconds are not equally distributed.
        This variable describes the maximum allowed overlap of scenes in seconds.

    Returns
    -------
    df_robots_first_appearences : pd.DataFrame
        columns = ['TIMESTAMP', 'node_id']
        Contains first appearences of nodes that last for at least scene_time seconds.
        Another requirement is that these nodes are spread by scene_time,
        because more than one ego node in a time window is redundant.

    If there are no potential robot nodes in the take return None.
    """
    # df : nodes that are vehicles
    df = df.groupby("type").get_group("VEHICLE").copy()

    # df : nodes that last the full scene time
    df_groups = df.groupby("node_id")
    total_time_per_obj = df_groups["TIMESTAMP"].apply(lambda x: x.max() - x.min())
    potential_robot_nodes = total_time_per_obj.loc[
        total_time_per_obj > scene_time
    ].index.values
    if len(potential_robot_nodes) == 0:
        return pd.DataFrame({})
    df_potential_robots = df.loc[df["node_id"].isin(potential_robot_nodes)]

    # df : first timesteps of overy node that lasts the full scene time
    gr_potential_robots = df_potential_robots.groupby("node_id")
    df_potential_robots_first_appearences = gr_potential_robots["TIMESTAMP"].apply(
        lambda x: x.min()
    )

    # df : make sure there is only one potential robot node per timestep
    gr_potential_robots_by_time = (
        df_potential_robots_first_appearences.to_frame()
        .reset_index(drop=False)
        .groupby("TIMESTAMP")
    )
    df_potential_robots_first_appearences = gr_potential_robots_by_time.apply(
        lambda x: x.iloc[0]
    ).reset_index(drop=True)

    # df : only take nodes that start 20 seconds apart to minimize scene overlap
    df_robots_first_appearences = pd.DataFrame({})
    last_scene_first_timestep = -(scene_time - max_scene_overlap)
    for _, row in df_potential_robots_first_appearences.iterrows():
        if (
            row["TIMESTAMP"]
            > last_scene_first_timestep + scene_time - max_scene_overlap
        ):
            last_scene_first_timestep = row["TIMESTAMP"]
            df_robots_first_appearences = df_robots_first_appearences.append(row)
    df_robots_first_appearences.reset_index(drop=True, inplace=True)

    # df : drop last scene because it might be too short
    last_row = len(df_robots_first_appearences) - 1
    if last_row != -1:
        df_robots_first_appearences.drop(index=[last_row], inplace=True)

    return df_robots_first_appearences


def extract_scene_from_df(df, timestamp, scene_time, dt, scene_id):
    """
    Arguments
    ---------
    df : pd.DataFrame
    timestamp : float
    scene_time : float
    dt : float
    scene_id : int

    Returns
    -------
    scene : pd.DataFrame
    """
    first_timestamp = timestamp
    last_timestamp = int(timestamp + scene_time)
    scene_idx = (df["TIMESTAMP"] >= first_timestamp) & (
        df["TIMESTAMP"] < last_timestamp
    )
    scene = df.loc[scene_idx].copy()
    scene["TIMESTAMP"] -= first_timestamp
    scene["scene_id"] = scene_id
    return scene


def set_robot_node(scene, node_id):
    """Replaces node_id with 'ego' and sets robot = True

    Arguments
    ---------
    scene : pd.DataFrame
        columns : ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading', 'scene_id']

    Returns
    -------
    scene : pd.DataFrame
        columns : ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading', 'scene_id']
    """
    robot_idx = scene["node_id"] == node_id
    scene.loc[robot_idx, "node_id"] = "ego"
    scene.loc[robot_idx, "robot"] = True
    return scene


def reset_node_id(scene):
    """Reset node_ids, because we don't track nodes between scenes

    Arguments
    --------
    scene : pd.DataFrame

    Returns
    -------
    scene : pd.DataFrame
    """
    not_ego_node_ids = np.array(
        [node_id for node_id in scene.node_id.unique() if node_id != "ego"]
    )
    not_ego_node_ids = np.sort(not_ego_node_ids)
    for new_node_id, node_id in enumerate(not_ego_node_ids):
        scene.loc[scene.node_id == node_id, "node_id"] = new_node_id
    scene.node_id = scene.node_id.astype(str)
    return scene


def set_frame_id(scene):
    """Marks every timestep with an integer index

    Arguments
    ---------
    scene : pd.DataFrame
        columns : ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading', 'scene_id']

    Returns
    -------
    scene : pd.DataFrame
        columns : ['TIMESTAMP', 'node_id', 'type', 'robot', 'x', 'y', 'heading', 'scene_id', 'frame_id']
    """
    scene.loc[:, "frame_id"] = int(0)
    for frame_id, timestamp in enumerate(scene.TIMESTAMP.unique()):
        scene.loc[scene.TIMESTAMP == timestamp, "frame_id"] = int(frame_id)
    return scene


###############################################################################
###   MAP PREPROCESSING   #####################################################
###############################################################################


def preprocess_openDD_map(map_path, map_name, args):

    ########################################
    # EXTRACT RAW MAP DATA
    ########################################
    df = load_map_df(map_path)
    (lane_df, border_df, lanes_boundaries) = get_lanes_from_map_df(df, dataformat="df")

    (
        drivable_area_arrays,
        non_drivable_area_arrays,
        areas_boundaries,
    ) = get_areas_from_map_df(df, dataformat="arrays")

    #########################################
    # BASIC PREPARATION
    #########################################
    map_boundaries = determine_total_map_boundaries(areas_boundaries, lanes_boundaries)
    map_boundaries_pixel = stretch_boundaries(map_boundaries, args.fraction)
    (x_min_map, x_max_map, y_min_map, y_max_map) = map_boundaries_pixel
    map_buffer_pixel = int(args.map_buffer / args.fraction)
    empty_map_mask_layer = np.zeros(
        (y_max_map - y_min_map + 1, x_max_map - x_min_map + 1)
    )

    ###########################################
    # CREATE LAYERS
    ###########################################

    # 1) LANE LAYER
    map_mask_lanes = create_map_mask_of_lanes(
        df=lane_df.loc[:, ["x", "y"]],
        map_boundaries_pixel=map_boundaries_pixel,
        fraction=args.fraction,
    )

    map_mask_lanes = thicken_lanes(map_mask_lanes, args.lane_radius, args.fraction)

    # 2) BOUNDARIE LAYER
    map_mask_border = create_map_mask_of_lanes(
        df=border_df.loc[:, ["x", "y"]],
        map_boundaries_pixel=map_boundaries_pixel,
        fraction=args.fraction,
    )

    # 3) AREA LAYERS
    map_mask_areas = empty_map_mask_layer.copy()
    areas_dict = {
        "drivable": drivable_area_arrays,
        "non_drivable": non_drivable_area_arrays,
    }
    for drivable_status, area_arrays in areas_dict.items():
        for polygon in area_arrays:
            area_mask, area_boundaries = create_area_mask_from_polygon(
                polygon, args.fraction
            )
            area_boundaries_pixel = stretch_boundaries(area_boundaries, args.fraction)
            map_mask_areas = insert_area_into_map_mask(
                map_boundaries_pixel,
                map_mask_areas,
                area_boundaries_pixel,
                area_mask,
                drivable=True if drivable_status == "drivable" else False,
            )
    map_mask_areas = map_mask_areas > 0

    ###########################################
    # CONCAT LAYERS
    ###########################################
    map_mask = np.stack((map_mask_lanes, map_mask_areas, map_mask_border), axis=0)

    ############################################
    # Expand Area besides original Area
    ############################################
    map_mask = add_padding_to_map(map_mask, map_buffer_pixel)
    map_mask = add_lanes_outside_map(map_mask)

    ###########################################
    # unite layers to "total_drivable_area"
    map_mask = map_mask[:2].max(axis=0, keepdims=True)
    data_map = DataMap.from_openDD(map_mask, map_boundaries, map_name, args)
    return data_map


def load_map_df(sql_path):
    dir, file = os.path.split(sql_path)
    table_name = file.split(".")[0].split("_")[1]
    con = sqlite3.connect(sql_path)
    df = pd.read_sql_query(f"SELECT * from {table_name}", con)
    con.close()
    return df


def get_lanes_from_map_df(df, dataformat):
    lanes_df = df[(df.type == "trafficLane") | (df.type == "boundary")][
        ["type", "geometry"]
    ].dropna()

    # prepare different output dataformats
    lanes_lists = []
    lanes_arrays = []
    lanes_dataframe = pd.DataFrame({"x": [], "y": []})

    bound_lists = []
    bound_arrays = []
    bound_dataframe = pd.DataFrame({"x": [], "y": []})

    # initialize star min/max values
    x_min_lanes = 1e8
    x_max_lanes = 0
    y_min_lanes = 1e8
    y_max_lanes = 0
    lanes_id = 0

    lanes_ids = range(len(lanes_df))
    for lane_id in lanes_ids:

        # convert string data to float
        linestring_list = []
        lane_type = lanes_df.type.iloc[lane_id]
        linestring_strings = (
            lanes_df.geometry.iloc[lane_id].split("(")[1].split(")")[0].split(",")
        )
        point_ids = range(len(linestring_strings))

        for point_id in point_ids:
            x = float(linestring_strings[point_id].split()[0])
            y = float(linestring_strings[point_id].split()[1])
            linestring_list.append((x, y))

        linestring_arr = np.array(linestring_list)

        linestring_df = pd.DataFrame(
            {"x": [linestring_arr[:, 0]], "y": [linestring_arr[:, 1]]}
        )
        # determine max and min
        x_min_linestring, y_min_linestring = tuple(linestring_arr.min(axis=0))
        x_max_linestring, y_max_linestring = tuple(linestring_arr.max(axis=0))

        if x_min_linestring < x_min_lanes:
            x_min_lanes = x_min_linestring
        if y_min_linestring < y_min_lanes:
            y_min_lanes = y_min_linestring
        if x_max_linestring > x_max_lanes:
            x_max_lanes = x_max_linestring
        if y_max_linestring > y_max_lanes:
            y_max_lanes = y_max_linestring

            # separate lane types
        if lane_type == "trafficLane":
            lanes_lists.append(linestring_list)
            lanes_arrays.append(linestring_arr)
            lanes_dataframe = lanes_dataframe.append(linestring_df)

        elif lane_type == "boundary":
            bound_lists.append(linestring_list)
            bound_arrays.append(linestring_arr)
            bound_dataframe = bound_dataframe.append(linestring_df)

    # determine output data format
    if dataformat == "arrays":
        lanes = lanes_arrays
        bound = bound_arrays

    elif dataformat == "lists":
        lanes = lanes_lists
        bound = bound_lists

    elif dataformat == "df":
        lanes = lanes_dataframe.reset_index(drop=True)
        bound = bound_dataframe.reset_index(drop=True)

    lanes_boundaries = x_min_lanes, x_max_lanes, y_min_lanes, y_max_lanes

    return lanes, bound, lanes_boundaries


def get_areas_from_map_df(df, dataformat):

    areas_df = df[df.type == "area"]

    # prepare different output dataformats
    drivable_areas_lists = []
    drivable_areas_arrays = []
    drivable_areas_dataframe = pd.DataFrame({"x": [], "y": []})

    non_drivable_areas_lists = []
    non_drivable_areas_arrays = []
    non_drivable_areas_dataframe = pd.DataFrame({"x": [], "y": []})

    # initialize star min/max values
    x_min_areas = 1e8
    x_max_areas = 0
    y_min_areas = 1e8
    y_max_areas = 0

    area_ids = range(len(areas_df))
    for area_id in area_ids:

        # convert string data to float
        polygon_list = list()
        area_type = areas_df.areaType.iloc[area_id]
        polygon_strings = (
            areas_df.geometry.iloc[area_id].split("((")[1].split("))")[0].split(",")
        )
        point_ids = range(len(polygon_strings))

        for point_id in point_ids:
            x = float(polygon_strings[point_id].split()[0])
            y = float(polygon_strings[point_id].split()[1])
            polygon_list.append((x, y))

        polygon_arr = np.array(polygon_list)
        polygon_df = pd.DataFrame({"x": [polygon_arr[:, 0]], "y": [polygon_arr[:, 1]]})

        # determine max and min
        x_min_polygon, y_min_polygon = tuple(polygon_arr.min(axis=0))
        x_max_polygon, y_max_polygon = tuple(polygon_arr.max(axis=0))

        if x_min_polygon < x_min_areas:
            x_min_areas = x_min_polygon
        if y_min_polygon < y_min_areas:
            y_min_areas = y_min_polygon
        if x_max_polygon > x_max_areas:
            x_max_areas = x_max_polygon
        if y_max_polygon > y_max_areas:
            y_max_areas = y_max_polygon

        # separate area types
        if area_type in ["COMPLETE"]:
            drivable_areas_lists.append(polygon_list)
            drivable_areas_arrays.append(polygon_arr)
            drivable_areas_dataframe = drivable_areas_dataframe.append(polygon_df)

        elif area_type in ["NEVER", "EMERGENCY"]:
            non_drivable_areas_lists.append(polygon_list)
            non_drivable_areas_arrays.append(polygon_arr)
            non_drivable_areas_dataframe = non_drivable_areas_dataframe.append(
                polygon_df
            )

    # determine output dataformat
    if dataformat == "arrays":
        drivable_areas = drivable_areas_arrays
        non_drivable_areas = non_drivable_areas_arrays

    elif dataformat == "lists":
        drivable_areas = drivable_areas_lists
        non_drivable_areas = non_drivable_areas_lists

    elif dataformat == "df":
        drivable_areas = drivable_areas_dataframe
        non_drivable_areas = non_drivable_areas_dataframe

    areas_boundaries = x_min_areas, x_max_areas, y_min_areas, y_max_areas

    return drivable_areas, non_drivable_areas, areas_boundaries


def create_map_mask_of_lanes(df, map_boundaries_pixel, fraction):

    df_lanes = interpolate_lanes_df_to_fraction(df, fraction)
    lanes_coordinates_array = unite_lane_segments(df_lanes)
    map_mask_lanes = create_mask_from_coordinates(
        lanes_coordinates_array, map_boundaries_pixel, fraction
    )
    return map_mask_lanes


def add_padding_to_map(map_mask, map_buffer_pixel):
    padded_mask = np.zeros(
        (
            map_mask.shape[0],
            map_mask.shape[1] + 2 * map_buffer_pixel,
            map_mask.shape[2] + 2 * map_buffer_pixel,
        )
    )
    padded_mask[
        :, map_buffer_pixel:-map_buffer_pixel, map_buffer_pixel:-map_buffer_pixel
    ] = map_mask
    return padded_mask


def add_lanes_outside_map(map_mask):
    M = map_mask[0].copy()
    y, x = np.argwhere(M).T
    edge_range = 4
    interval_l = x.min() + edge_range
    interval_r = M.shape[1] - x.max() + edge_range
    interval_b = y.min() + edge_range
    interval_t = M.shape[0] - y.max() + edge_range

    bottom = M[:interval_b, :]
    top = M[-interval_t:, :]
    left = M[:, :interval_l]
    right = M[:, -interval_r:]

    y_b, x_b = np.argwhere(bottom).T
    y_t, x_t = np.argwhere(top).T
    y_l, x_l = np.argwhere(left).T
    y_r, x_r = np.argwhere(right).T

    M[:interval_b, x_b.min() : x_b.max()] = 1
    M[-interval_t:, x_t.min() : x_t.max()] = 1
    M[y_l.min() : y_l.max(), :interval_l] = 1
    M[y_r.min() : y_r.max(), -interval_r:] = 1
    map_mask[0] = M
    return map_mask


def interpolate_lanes_df_to_fraction(df, fraction):
    df_interpolated = df.apply(
        lambda row: interpolate_lane_segment_to_fraction(row, fraction),
        axis=1,
        result_type="expand",
    )
    df_interpolated.columns = ["x", "y"]
    return df_interpolated


def interpolate_lane_segment_to_fraction(row, fraction):
    x = row.x
    y = row.y

    y_min = round_to_fraction(y.min(), fraction)
    y_max = round_to_fraction(y.max(), fraction)
    x_min = round_to_fraction(x.min(), fraction)
    x_max = round_to_fraction(x.max(), fraction)

    y_is_const = y_max == y_min
    x_is_const = x_max == x_min

    y_right_order = np.all(np.diff(y) > 0)
    x_right_order = np.all(np.diff(x) > 0)

    if swapping_axes_is_necessary_bool(x):
        if not y_right_order:
            x = np.flip(x)
            y = np.flip(y)
        if y_is_const:
            x_interp = np.arange(x_min, x_max, 1 / 3)
            y_interp = y_max * np.ones(len(x_interp))
        else:
            y_interp = np.arange(y_min, y_max, 1 / 3)
            x_interp = round_to_fraction(np.interp(y_interp, y, x), fraction)

    else:
        if not x_right_order:
            x = np.flip(x)
            y = np.flip(y)
        if x_is_const:
            y_interp = np.arange(y_min, y_max, 1 / 3)
            x_interp = x_max * np.ones(len(y_interp))
        else:
            x_interp = np.arange(x_min, x_max, 1 / 3)
            y_interp = round_to_fraction(np.interp(x_interp, x, y), fraction)

    return [x_interp, y_interp]


def swapping_axes_is_necessary_bool(x):
    idx_x_loc_min = argrelextrema(x, np.less)[0]
    idx_x_loc_max = argrelextrema(x, np.greater)[0]

    optimum_exists = not (len(idx_x_loc_min) == 0 and len(idx_x_loc_max) == 0)

    if optimum_exists:
        swap_axes = True
    else:
        swap_axes = False

    return swap_axes


def unite_lane_segments(df):
    x = df.x.values
    y = df.y.values
    x_vec = x[0]
    y_vec = y[0]
    for lane_segment in range(1, len(df)):
        x_vec = np.concatenate((x_vec, x[lane_segment]), axis=0)
        y_vec = np.concatenate((y_vec, y[lane_segment]), axis=0)
    return x_vec, y_vec


def create_mask_from_coordinates(coordinates, map_boundaries_pixel, fraction):
    (x, y) = stretch_boundaries(coordinates, fraction)
    (x_min, x_max, y_min, y_max) = map_boundaries_pixel

    x_normalized = x - x_min
    y_normalized = y - y_min

    canvas_shape = (y_max - y_min + 1, x_max - x_min + 1)
    mask = np.zeros(canvas_shape)

    for i in range(len(x_normalized)):
        position = (y_normalized[i], x_normalized[i])
        mask[position] = 1

    return mask


def thicken_lanes(mask, radius, fraction):
    # add padding
    stretched_radius = int(radius / fraction)
    padding = stretched_radius
    padded_mask_shape = (mask.shape[0] + 2 * padding, mask.shape[1] + 2 * padding)
    padded_mask = np.zeros(padded_mask_shape)
    padded_mask[padding:-padding, padding:-padding] = mask

    # find True mask entry
    for y in range(padding, padded_mask.shape[0] - padding):
        for x in range(padding, padded_mask.shape[1] - padding):
            if padded_mask[y, x] == 1:

                # iterate over square around position
                for distance_y in range(-stretched_radius, stretched_radius + 1):
                    for distance_x in range(-stretched_radius, stretched_radius + 1):

                        relative_position = np.array([distance_x, distance_y])
                        relative_distance = np.linalg.norm(relative_position)

                        # find points, that are within a circle
                        if relative_distance <= stretched_radius:
                            padded_mask[y + distance_y, x + distance_x] = 2

    # remove padding
    new_mask = padded_mask[padding:-padding, padding:-padding]
    return new_mask / 2


def stretch_boundaries(boundaries, fraction):
    stretched_boundaries = tuple(
        (round_to_fraction(np.array(boundaries), fraction) / fraction).astype(int)
    )
    return stretched_boundaries


def round_to_fraction(x, fraction):
    return np.round(x / fraction) * fraction


def determine_total_map_boundaries(areas_boundaries, lanes_boundaries):

    x_min_lanes, x_max_lanes, y_min_lanes, y_max_lanes = lanes_boundaries
    x_min_areas, x_max_areas, y_min_areas, y_max_areas = areas_boundaries
    x_min = min(x_min_areas, x_min_lanes)
    x_max = max(x_max_areas, x_max_lanes)
    y_min = min(y_min_areas, y_min_lanes)
    y_max = max(y_max_areas, y_max_lanes)

    map_boundaries = x_min, x_max, y_min, y_max
    return map_boundaries


def create_area_mask_from_polygon(polygon, fraction):
    # ACHTUNG: für 'Path' ist y die erste koordinate, deswegen flip()
    polygon_arr = np.array(polygon)
    polygon_arr = np.flip(polygon_arr, axis=1)

    polygon_min = polygon_arr.min(axis=0)
    polygon_max = polygon_arr.max(axis=0)
    polygon_size = polygon_max - polygon_min
    polygon_size_stretched = stretch_boundaries(polygon_size, fraction)

    area_boundaries = polygon_min[1], polygon_max[1], polygon_min[0], polygon_max[0]

    polygon_arr_norm = polygon_arr - polygon_min
    polygon_arr_norm_stretched = (
        round_to_fraction(polygon_arr_norm, fraction) / fraction
    )

    poly_path = Path(polygon_arr_norm_stretched)

    height, width = polygon_size_stretched
    x, y = np.mgrid[:height, :width]
    coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    mask_flatten = poly_path.contains_points(coors)
    mask = mask_flatten.reshape(height, width)

    return mask, area_boundaries


def insert_area_into_map_mask(
    map_boundaries_pixel, map_mask, area_boundaries_pixel, area_mask, drivable
):
    x_min_area, x_max_area, y_min_area, y_max_area = area_boundaries_pixel
    x_min_map, x_max_map, y_min_map, y_max_map = map_boundaries_pixel

    x_min_area_rel = x_min_area - x_min_map
    x_max_area_rel = x_min_area_rel + area_mask.shape[1]
    y_min_area_rel = y_min_area - y_min_map
    y_max_area_rel = y_min_area_rel + area_mask.shape[0]

    if drivable:
        map_mask[y_min_area_rel:y_max_area_rel, x_min_area_rel:x_max_area_rel] += (
            1 * area_mask
        )
    elif not drivable:
        map_mask[y_min_area_rel:y_max_area_rel, x_min_area_rel:x_max_area_rel] -= (
            5 * area_mask
        )

    return map_mask
