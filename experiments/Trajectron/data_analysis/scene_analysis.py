"""
Module with functions for Analysing the "scene" and "node" class from the Trajectron++ "env"
"""
import numpy as np
import pandas as pd
from copy import deepcopy

DT = 0.5
MAX_STANDING_SPEED = 0.5
HIST_LEN = 4
PRED_HORIZON = 12
EPS = 0.00001
LOW_DISTANCE = 30
HIGH_DISTANCE = 60
LOW_ACCELERATION = 0.5
HIGH_ACCELERATION = 6
NO_TURNING_ANGLE = 15
LOW_TURNING_ANGLE = 45
MID_TURNING_ANGLE = 75
ROBOT_INTERACTION_DISTANCE = 20
M_PER_SECOND_TO_KM_PER_HOUR = 3.6


def node_curvature_change(node):
    """
    Returns True if nodes changes its direction of steering

    :param node: object of Node class
    :return: curvature_change_bool
    """
    x = node.data.data[:, :2]

    def angle(v0, v1):
        return (
            np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)) * 360 / (2 * np.pi)
        )

    steps = np.diff(x, axis=0)
    angles = [angle(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]
    if len(angles) < 2:
        return False
    angle_change = np.stack([angles[i] * angles[i + 1] for i in range(len(angles) - 1)])
    curvature_change_bool = angle_change[(angle_change < -8)].any()
    not_random_fluctuation = not no_turning_angle(node)
    return curvature_change_bool and not_random_fluctuation


def node_stands_on_some_timestep(node):
    """
    Returns True if node stands on some timesteps

    :param node: object of Node class
    :return:
    """
    d_s = np.linalg.norm(np.diff(node.data.data[:, :2], axis=0), axis=1)
    standing_on_any_timestep_bool = (d_s < MAX_STANDING_SPEED * DT).any()
    return standing_on_any_timestep_bool


def node_stands_on_all_timesteps(node):
    """
    Returns True if node stands on all timesteps

    :param node: object of Node class
    :return:
    """
    d_s = np.linalg.norm(np.diff(node.data.data[:, :2], axis=0), axis=1)
    standing_on_all_timestep = (d_s < MAX_STANDING_SPEED * DT).all()
    return standing_on_all_timestep


def node_travelled_distance(node):
    """
    Maximum distance travelled by node within prediction horizon

    :param node: object of Node class
    :return:
    """
    xy = node.data.data[:, :2]
    distances = np.stack(
        [
            np.linalg.norm(xy[i] - xy[i + PRED_HORIZON])
            for i in range(len(xy) - PRED_HORIZON)
        ]
    )
    return distances.max()


def node_tangential_acceleration(node):
    """
    Maximum tangential acceleration of node

    :param node: object of Node class
    :return:
    """
    v_vec = np.diff(node.data.data[:, :2], axis=0) / DT
    v_norm = np.abs(np.linalg.norm(v_vec, axis=1)) + EPS
    a_vec = np.diff(v_vec, axis=0) / DT
    a_norm = np.abs(np.linalg.norm(a_vec, axis=1)) + EPS
    if len(a_norm) == 0:
        return 0
    direction = v_vec / v_norm.reshape(-1, 1)
    a_tangential = np.stack(
        [(direction[ts] @ a_vec[ts]) for ts in range(PRED_HORIZON - 2)], axis=0
    )
    return a_tangential.mean()


def node_max_turning_angle(node):
    """
    Maximum change of course of the node

    :param node: object of Node class
    :return:
    """
    xy = node.data.data[:, :2]
    d_s = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    v_vec = np.diff(xy, axis=0) / DT
    v_norm = np.linalg.norm(v_vec, axis=1) + EPS
    direction = v_vec / v_norm.reshape(-1, 1)

    idx_moving = np.argwhere(d_s > MAX_STANDING_SPEED * DT).flatten()
    direction_moving = direction[idx_moving]
    alphas = []
    for i in range(len(direction_moving - PRED_HORIZON)):
        current_direction = direction_moving[i : i + PRED_HORIZON]
        cos_alpha = (current_direction[0] @ current_direction[-1]) / (
            np.linalg.norm(current_direction[0]) * np.linalg.norm(current_direction[-1])
        )
        cos_alpha = cos_alpha.clip(min=-1).clip(max=1)
        alpha = np.abs(np.arccos(cos_alpha) * 360 / (2 * np.pi))
        alphas.append(alpha)
    if len(alphas) == 0:
        return 0
    return np.array(alphas).max().item()


def node_max_velocity(node):
    """
    Maximum velocity of a node

    :param node: object of Node class
    :return:
    """
    v = node.get(tr_scene=np.array([0, 39]), state={"velocity": ["x", "y"]})
    not_nan_idx = list(set(list(np.where(~np.isnan(v))[0])))
    v_not_nan = v[not_nan_idx]
    if len(v_not_nan) == 0:
        v_max = None
    else:
        v_max = np.linalg.norm(v_not_nan, axis=1).max() * M_PER_SECOND_TO_KM_PER_HOUR
    return v_max


def node_stand(node):
    """
    Return True if node stands on all timesteps

    :param node: object of Node class
    :return:
    """
    return node_stands_on_all_timesteps(node)


def node_start(node):
    """
    Return True if node stands on some timesteps, but not on all

    :param node: object of Node class
    :return:
    """
    return node_stands_on_some_timestep(node) and not node_stands_on_all_timesteps(node)


def node_move(node):
    """
    Return True if node moves on all timesteps

    :param node: object of Node class
    :return:
    """
    return not node_stands_on_some_timestep(node)


def low_distance(node):
    """
    Return True if node travels a low distance within the prediction horizon

    :param node: object of Node class
    :return:
    """
    return node_travelled_distance(node) < LOW_DISTANCE


def mid_distance(node):
    """
    Return True if node travels a middle distance within the prediction horizon

    :param node: object of Node class
    :return:
    """
    return (
        node_travelled_distance(node) > LOW_DISTANCE
        and node_travelled_distance(node) < HIGH_DISTANCE
    )


def high_distance(node):
    """
    Return True if node travells a high distance within the prediction horizon

    :param node: object of Node class
    :return:
    """
    return node_travelled_distance(node) > HIGH_DISTANCE


def node_no_acceleration(node):
    """
    Return True if node has no acceleration

    :param node: object of Node class
    :return:
    """
    return (
        node_tangential_acceleration(node) < LOW_ACCELERATION
        and node_tangential_acceleration(node) > -LOW_ACCELERATION
    )


def node_acceleration(node):
    """
    Return True if node has a medium acceleration

    :param node: object of Node class
    :return:
    """
    return (
        node_tangential_acceleration(node) > LOW_ACCELERATION
        and node_tangential_acceleration(node) < HIGH_ACCELERATION
    )


def node_negative_acceleration(node):
    """
    Return True if node has a medium acceleration

    :param node: object of Node class
    :return:
    """
    return node_tangential_acceleration(node) < -LOW_ACCELERATION


def node_high_acceleration(node):
    """
    Return True if node has a high acceleration

    :param node: object of Node class
    :return:
    """
    return node_tangential_acceleration(node) > HIGH_ACCELERATION


def no_turning_angle(node):
    """
    Return True if node does no change of course

    :param node: object of Node class
    :return:
    """
    return node_max_turning_angle(node) < NO_TURNING_ANGLE


def turning_angle(node):
    """
    Return True if node changes course at all

    :param node: object of Node class
    :return:
    """
    return node_max_turning_angle(node) > NO_TURNING_ANGLE


def low_turning_angle(node):
    """
    Return True if node changes course to a low degree

    :param node: object of Node class
    :return:
    """
    return (
        node_max_turning_angle(node) > NO_TURNING_ANGLE
        and node_max_turning_angle(node) < LOW_TURNING_ANGLE
    )


def mid_turning_angle(node):
    """
    Return True if node changes course to a middle degree

    :param node: object of Node class
    :return:
    """
    return (
        node_max_turning_angle(node) > LOW_TURNING_ANGLE
        and node_max_turning_angle(node) < MID_TURNING_ANGLE
    )


def high_turning_angle(node):
    """
    Return True if node changes course to a high degree

    :param node: object of Node class
    :return:
    """
    return node_max_turning_angle(node) > MID_TURNING_ANGLE


def node_interactions(node, scene):
    """
    Number of interactions of a node with other not standing nodes within a scene-

    :param node:
    :param scene:
    :return:
    """
    tsg = scene.temporal_scene_graph
    my_node_idx = tsg.node_index_lookup[node]
    neighbours_idx = [
        idx
        for node, idx in tsg.node_index_lookup.items()
        if not node_stands_on_all_timesteps(node)
    ]
    num_interactions = np.tril(tsg.adj_mat)[my_node_idx, neighbours_idx].sum()
    return num_interactions


def robot_interactions(node, scene):
    """
    Return True if node interacts with robot

    :param node:
    :param scene:
    :return:
    """
    xy_node = node.data.data[:, :2]
    xy_robot = scene.robot.data.data[:, :2]

    first = node.first_timestep
    last = node._last_timestep
    distance_to_robot = np.linalg.norm(
        xy_robot[first : last + 1] - xy_node, axis=1
    ).min()
    return distance_to_robot < ROBOT_INTERACTION_DISTANCE


def scene_interaction_density(scene):
    """
    Number of total Interactions in a scene per second

    :param scene:
    :return:
    """
    tsg = scene.temporal_scene_graph
    vehicle_indices = [
        idx
        for node, idx in tsg.node_index_lookup.items()
        if (str(node.type) == "VEHICLE" and not node_stands_on_all_timesteps(node))
    ]
    edges_per_sec = np.tril(tsg.adj_mat)[vehicle_indices].sum() / (scene.timesteps * DT)
    return edges_per_sec


######################################################################
#### SCENE ANALYSIS  #################################################
######################################################################


def scene_driving_characteristics(scene):
    """
    Return dict with driving characteristics of the scene

    :param scene:
    :return:
    """
    scene_driving_characteristics = {
        "Bewegungsart": {"stand": 0, "start": 0, "move": 0},
        "Beschleunigung": {"const": 0, "acc": 0, "dec": 0},
        "Kursaenderung": {"00_15": 0, "15_45": 0, "45_75": 0, "75_360": 0},
        "Lenkrichtung": {"keep": 0, "change": 0},
        "Distanz": {"short": 0, "middle": 0, "long": 0},
        "Interaktion": {"n2n": 0, "n2r": 0},
    }
    for node in scene.nodes:
        if node._last_timestep > scene.robot._last_timestep:
            continue
        if node.is_robot and not scene.timesteps == 51:
            continue
        if not str(node.type) == "VEHICLE":
            continue
        if not (len(node.data.data) >= HIST_LEN + PRED_HORIZON + 1):
            continue

        scene_driving_characteristics["Bewegungsart"]["move"] += node_move(node)
        scene_driving_characteristics["Bewegungsart"]["start"] += node_start(node)
        scene_driving_characteristics["Bewegungsart"]["stand"] += node_stand(node)

        if node_move(node):
            scene_driving_characteristics["Beschleunigung"][
                "const"
            ] += node_no_acceleration(node)
            scene_driving_characteristics["Beschleunigung"]["acc"] += node_acceleration(
                node
            )
            scene_driving_characteristics["Beschleunigung"][
                "dec"
            ] += node_negative_acceleration(node)

            scene_driving_characteristics["Kursaenderung"]["00_15"] += no_turning_angle(
                node
            )
            scene_driving_characteristics["Kursaenderung"][
                "15_45"
            ] += low_turning_angle(node)
            scene_driving_characteristics["Kursaenderung"][
                "45_75"
            ] += mid_turning_angle(node)
            scene_driving_characteristics["Kursaenderung"][
                "75_360"
            ] += high_turning_angle(node)

            scene_driving_characteristics["Lenkrichtung"][
                "change"
            ] += node_curvature_change(node)
            scene_driving_characteristics["Lenkrichtung"][
                "keep"
            ] += not node_curvature_change(node)

            scene_driving_characteristics["Distanz"]["short"] += low_distance(node)
            scene_driving_characteristics["Distanz"]["middle"] += mid_distance(node)
            scene_driving_characteristics["Distanz"]["long"] += high_distance(node)

            scene_driving_characteristics["Interaktion"]["n2n"] += node_interactions(
                node, scene
            )
            scene_driving_characteristics["Interaktion"]["n2r"] += robot_interactions(
                node, scene
            )
    return scene_driving_characteristics


##########################################################################
### CONVERTER  ###########################################################
##########################################################################


def scene_velocity_stats(scene):
    """Scene Velocity statistics in km/h
    Arguments
    ---------
    scene : Scene

    Returns
    -------
    velocity_stats : dict()
    """
    velocities = []
    for node in scene.nodes:
        if str(node.type) == "PEDESTRIAN":
            continue
        v_max = node_max_velocity(node)
        if v_max is None:
            continue
        velocities.append(v_max)
    velocities = np.sort(np.array(velocities))
    velocity_stats = {
        "mean": velocities.mean(),
        "max": velocities[-1].item(),
        "min": velocities[0].item(),
    }
    return velocity_stats


def scene_stats_dict2df(scene_stats_dict):
    """
    convert dict with scene information to pd.dataframe

    :param scene_stats_dict:
    :return:
    """
    df_scene_stats = pd.DataFrame()
    for high_key, item in scene_stats_dict.items():
        for low_key, low_item in item.items():
            df = pd.DataFrame({(high_key, low_key): [low_item]})
            df.columns.names = ["Fahrcharakteristik", "Kategorie"]
            df_scene_stats = pd.concat([df_scene_stats, df], axis=1)
    return df_scene_stats


def env_stats(env):
    """
    Get the driving characteristics of an environment

    :param env:
    :return:
    """
    df_env_stats = pd.DataFrame()
    for scene in env.scenes:
        df_env_stats = df_env_stats.append(
            scene_stats_dict2df(scene_driving_characteristics(scene))
        )
    df_env_stats.reset_index(drop=True)

    df_abs = df_env_stats.sum().to_frame().T
    df_rel = deepcopy(df_abs)

    total = df_abs["Bewegungsart"].sum(axis=1).values.item()
    move = df_abs[("Bewegungsart", "move")].values.item()

    df_rel["Bewegungsart"] = df_rel["Bewegungsart"] / total * 100
    move_cols = [
        col
        for col in df_rel.columns.get_level_values("Fahrcharakteristik").unique()
        if not col == "Bewegungsart"
    ]
    df_rel[move_cols] = (df_rel[move_cols] / move * 100).values
    return df_rel


def split_scene_by_condition(scene, and_conditions, bools):
    """
    Return 2 scene objects.
    scene_true:  contains only nodes, that satisfy all conditions.
    scene_false:  contains only nodes, that dont satisfy all conditions.

    :param scene:
    :param and_conditions: list of boolean functions with node obj as input
    :param bools: expected boolean values of conditions
    :return:
    """
    nodes_true = []
    nodes_false = []
    scene_true = deepcopy(scene)
    scene_false = deepcopy(scene)

    for node in scene.nodes:
        if not (len(node.data.data) >= HIST_LEN + PRED_HORIZON + 1):
            continue
        if np.array(
            [condition(node) == bools[i] for i, condition in enumerate(and_conditions)]
        ).all():
            nodes_true.append(node)
        else:
            nodes_false.append(node)

    scene_true.nodes = nodes_true
    scene_false.nodes = nodes_false

    return scene_true, scene_false


def split_env_by_condition(env, and_conditions, bools):
    """
    Return 2 env objects.
    env_true:  contains only nodes, that satisfy all conditions.
    env_false:  contains only nodes, that dont satisfy all conditions.

    :param scene:
    :param and_conditions: list of boolean functions with node obj as input
    :param bools: expected boolean values of conditions
    :return:
    """
    scenes_true = []
    scenes_false = []
    env_true = deepcopy(env)
    env_false = deepcopy(env)

    for scene in env.scenes:
        scene_true, scene_false = split_scene_by_condition(scene, and_conditions, bools)
        scenes_true.append(scene_true)
        scenes_false.append(scene_false)

    env_true.scenes = scenes_true
    env_false.scenes = scenes_false
    return env_true, env_false
