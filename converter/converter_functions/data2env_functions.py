"""
Module that contains functions for the processing from "data" format to "env" format
"""
import os
import sys
import random
import dill
import numpy as np
import pandas as pd
from kalman_filter import NonlinearKinematicBicycle

sys.path.append("../../Trajectron/trajectron")
from environment import Environment, Scene, Node, GeometricMap, derivative_of
from preprocessing_utils import ensure_dir

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

data_columns_vehicle = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration", "heading"], ["x", "y"]]
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")])
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]])
)
data_columns_pedestrian = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]]
)
standardization = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    },
    "VEHICLE": {
        "position": {"x": {"mean": 0, "std": 80}, "y": {"mean": 0, "std": 80}},
        "velocity": {
            "x": {"mean": 0, "std": 15},
            "y": {"mean": 0, "std": 15},
            "norm": {"mean": 0, "std": 15},
        },
        "acceleration": {
            "x": {"mean": 0, "std": 4},
            "y": {"mean": 0, "std": 4},
            "norm": {"mean": 0, "std": 4},
        },
        "heading": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
            "°": {"mean": 0, "std": np.pi},
            "d°": {"mean": 0, "std": 1},
        },
    },
}
node_type_list = ["VEHICLE", "PEDESTRIAN"]

CURV_0_2 = 0
CURV_0_1 = 0
TOTAL_CURV = 0


def shift_data(data, buffer):
    """
    shift coordinates of all vehicles so that minimal value is the buffer value
    """
    x_min_scene = data["x"].min() - buffer
    y_min_scene = data["y"].min() - buffer
    x_max_scene = data["x"].max() + buffer
    y_max_scene = data["y"].max() + buffer
    scene_boundaries = x_min_scene, x_max_scene, y_min_scene, y_max_scene

    data["x"] -= x_min_scene
    data["y"] -= y_min_scene
    return data, scene_boundaries


def data_map2env_map(scene_mask, fraction=1 / 3):
    """
    Arguments
    ---------
    scene_mask : np.array[1,x,y] or np.array[x,y]

    Returns
    -------
    type_map : dict(GeometricMap)
    """
    homography = np.eye(3) / fraction
    if len(scene_mask.shape) == 2:
        scene_mask = np.expand_dims(scene_mask, axis=0)

    scene_mask = np.swapaxes(scene_mask, 1, 2)
    scene_mask = (scene_mask * 255.0).astype(np.uint8)
    geo_map = GeometricMap(
        data=scene_mask, homography=homography, description=", ".join(["drivable area"])
    )
    type_map = {
        geo_map_name: geo_map
        for geo_map_name in ["PEDESTRIAN", "VEHICLE", "VISUALIZATION"]
    }
    return type_map


def extract_nodes_from_data(data, scene, env):
    """
    create list of node objects of Node class from scene of "data" format
    """
    scene_nodes = []
    for node_id in pd.unique(data["node_id"]):
        node_frequency_multiplier = 1
        node_df = data[data["node_id"] == node_id]

        if node_df["x"].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df["frame_id"]) == 1):
            print('Occlusion')
            continue

        node_values = node_df[["x", "y"]].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df["heading"].values
        if node_df.iloc[0]["type"] == "VEHICLE" and not node_id == "ego":
            # Kalman filter Agent
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(x):
                    # assign new est values
                    x[i] = x_vec_est_new[0][0]
                    y[i] = x_vec_est_new[1][0]
                    heading[i] = x_vec_est_new[2][0]
                    velocity[i] = x_vec_est_new[3][0]

                if i < len(x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[x[i]], [y[i]], [heading[i]], [velocity[i]]])
                    z_new = np.array(
                        [[x[i + 1]], [y[i + 1]], [heading[i + 1]], [velocity[i + 1]]]
                    )
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.0], [0.0]]),
                        P_matrix=P_matrix,
                        z_new=z_new,
                    )
                    P_matrix = P_matrix_new

            curvature, path_length, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            if path_length < 1.0:  # vehicle is "not" moving
                x = x[0].repeat(scene.timesteps)  # 40
                y = y[0].repeat(scene.timesteps)  # 40
                heading = heading[0].repeat(scene.timesteps)  # 40
            global TOTAL_CURV
            global CURV_0_2
            global CURV_0_1
            TOTAL_CURV += 1
            if path_length > 1.0:
                if curvature > 0.2:
                    CURV_0_2 += 1
                    node_frequency_multiplier = 3 * int(np.floor(TOTAL_CURV / CURV_0_2))
                elif curvature > 0.1:
                    CURV_0_1 += 1
                    node_frequency_multiplier = 3 * int(np.floor(TOTAL_CURV / CURV_0_1))

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]["type"] == "VEHICLE":
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("velocity", "norm"): np.linalg.norm(
                    np.stack((vx, vy), axis=-1), axis=-1
                ),
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
                ("acceleration", "norm"): np.linalg.norm(
                    np.stack((ax, ay), axis=-1), axis=-1
                ),
                ("heading", "x"): heading_x,
                ("heading", "y"): heading_y,
                ("heading", "°"): heading,
                ("heading", "d°"): derivative_of(heading, scene.dt, radian=True),
            }
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        elif node_df.iloc[0]["type"] == "PEDESTRIAN":
            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            }
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        if node_df.iloc[0]["type"] == "VEHICLE":
            node_type = env.NodeType.VEHICLE
        elif node_df.iloc[0]["type"] == "PEDESTRIAN":
            node_type = env.NodeType.PEDESTRIAN
        node = Node(
            node_type=node_type,
            node_id=node_id,
            data=node_data,
            frequency_multiplier=node_frequency_multiplier,
        )

        node.first_timestep = node_df["frame_id"].iloc[0]
        if node_df.iloc[0]["robot"]:
            node.is_robot = True
            scene.robot = node
        if not np.isnan(node.get(np.array([0, 39]), {"position": ["x", "y"]})).all():
            scene_nodes.append(node)
    return scene_nodes


def process_scene(env, data, data_map, dt):
    """
    convert scene of "data" format to scene object of "env" format
    """
    data, scene_boundaries = shift_data(data, buffer=data_map.scene_buffer)
    scene_mask = data_map.extract_scene_mask_from_map_mask(scene_boundaries)

    scene = Scene(
        timesteps=data["frame_id"].max() + 1,
        dt=dt,
        name=str(data.loc[0, "scene_id"]),
        aug_func=augment,
    )
    scene.map = data_map2env_map(scene_mask)
    scene.nodes = extract_nodes_from_data(data, scene, env)
    return scene


def augment_scene(scene, angle):
    """
    create augmented version of scene object of Scene class
    """
    def rotate_pc(pc, alpha):
        """
        rotate xy-vector by alpha
        """
        M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration", "heading"], ["x", "y"]]
    )
    data_columns_vehicle = data_columns_vehicle.append(
        pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")])
    )
    data_columns_vehicle = data_columns_vehicle.append(
        pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]])
    )

    data_columns_pedestrian = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    scene_aug = Scene(
        timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene
    )

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == "PEDESTRIAN":
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            }

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(
                node_type=node.type,
                node_id=node.id,
                data=node_data,
                first_timestep=node.first_timestep,
            )
        elif node.type == "VEHICLE":
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, "°").copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("velocity", "norm"): np.linalg.norm(
                    np.stack((vx, vy), axis=-1), axis=-1
                ),
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
                ("acceleration", "norm"): np.linalg.norm(
                    np.stack((ax, ay), axis=-1), axis=-1
                ),
                ("heading", "x"): heading_x,
                ("heading", "y"): heading_y,
                ("heading", "°"): heading,
                ("heading", "d°"): derivative_of(heading, scene.dt, radian=True),
            }

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(
                node_type=node.type,
                node_id=node.id,
                data=node_data,
                first_timestep=node.first_timestep,
                non_aug_node=node,
            )

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    """
    augment function for scene object of Scene class
    """
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def trajectory_curvature(t):
    """
    calculate length of trajectory divided by distance of trajectory

    :param t: xy-coordinates of trajectory (np.array)
    :return: curvature (int)
    """
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.0):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def train_test_val_split(data, shuffle=True, split=(0.6, 0.2, 0.2)):
    """
    split dataframe into train, test, val
    """
    if shuffle:
        random.shuffle(data)

    train_split, val_split, test_split = split

    train_size = int(len(data) * train_split)
    val_size = int(len(data) * val_split)
    test_size = int(len(data) * test_split)

    train_data = data[:train_size]
    val_data = data[train_size: train_size + val_size]
    test_data = data[-test_size:]

    data_dict = {"train": train_data, "val": val_data, "test": test_data}
    return data_dict


def datadict2scenes(env, traj_dict, map_dict, split):
    """
    convert dict of "data" format scenes to list of objects of Scene class
    """
    scenes = []
    num_scenes = len([scene for take, _ in traj_dict[split] for scene in take])
    for traj_take, map_name in traj_dict[split]:
        for data in traj_take:
            scene = process_scene(env, data, map_dict[map_name], dt=0.5)
            if split == "train":
                augmented_scenes = []
                for angle in np.arange(0, 360, 15):
                    augmented_scenes.append(augment_scene(scene, angle))
                scene.augmented = augmented_scenes
            scenes.append(scene)
            sys.stdout.write(
                "\rProcessed {}/{} {} scenes.".format(len(scenes), num_scenes, split)
            )
    return scenes


def init_env():
    """
    create empty env object of the Environment class
    """
    env = Environment(
        node_type_list=["VEHICLE", "PEDESTRIAN"], standardization=standardization
    )
    attention_radius = {}
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE
    return env


def data2env(output_dir, traj_dict, map_dict, speed=None):
    """
    Transform dict of scenes in the "data" format to "env" format and write to file.
    """
    dataset = os.path.basename(output_dir).split("_")[0]
    for split in ["train", "test", "val"]:
        # Environment Setup
        env = init_env()
        env.scenes = datadict2scenes(
            env, traj_dict, map_dict, split
        )

        # write env with scenes into file
        if speed is None:
            data_dict_path = os.path.join(output_dir, f"{dataset}_{split}.pkl")
        else:
            data_dict_path = os.path.join(output_dir, f"{dataset}_{speed}_{split}.pkl")

        ensure_dir(output_dir)
        with open(data_dict_path, "wb") as file:
            dill.dump(env, file, protocol=dill.HIGHEST_PROTOCOL)
        print(f"Saved {split}-environment.\n")

        global TOTAL_CURV
        global CURV_0_2
        global CURV_0_1
        TOTAL_CURV = 0
        CURV_0_1 = 0
        CURV_0_2 = 0
