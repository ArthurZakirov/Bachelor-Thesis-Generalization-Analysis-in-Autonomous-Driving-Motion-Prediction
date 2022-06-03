"""
Module that contains functions for the processing from nuScenes raw to "data" format
"""
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split
from nuscenes.utils.splits import create_splits_scenes


def split_scene_names(version, val_split):
    """
    create train val split
    """
    splits = create_splits_scenes()
    train_scenes, val_scenes = train_test_split(
        splits["train" if "mini" not in version else "mini_train"], test_size=val_split
    )
    train_scene_names = splits["train" if "mini" not in version else "mini_train"]
    val_scene_names = val_scenes
    test_scene_names = splits["val" if "mini" not in version else "mini_val"]

    ns_scene_names = {}
    ns_scene_names["train"] = train_scene_names
    ns_scene_names["val"] = val_scene_names
    ns_scene_names["test"] = test_scene_names
    return ns_scene_names


def nuScenes_category2env_category(old_category, attribute, env):
    """
    adjust names of agent classes to "env" format
    """
    skip_bool = False
    ego_category = env.NodeType.VEHICLE
    if (
        "pedestrian" in old_category
        and not "stroller" in old_category
        and not "wheelchair" in old_category
    ):
        new_category = env.NodeType.PEDESTRIAN
    elif (
        "vehicle" in old_category
        and "bicycle" not in old_category
        and "motorcycle" not in old_category
        and "parked" not in attribute
    ):
        new_category = env.NodeType.VEHICLE
    else:
        new_category = None
        skip_bool = True
    return skip_bool, new_category, ego_category


def ns_scene2data(nusc, ns_scene, scene_id, env, node_selector_fun):
    """
    convert object of nuScenes Scene class to "data" format
    """
    data = pd.DataFrame(
        columns=[
            "frame_id",
            "type",
            "node_id",
            "robot",
            "x",
            "y",
            "z",
            "length",
            "width",
            "height",
            "heading",
        ]
    )

    sample_token = ns_scene["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    frame_id = 0

    while sample["next"]:
        annotation_tokens = sample["anns"]
        for annotation_token in annotation_tokens:
            annotation = nusc.get("sample_annotation", annotation_token)
            category = annotation["category_name"]
            if len(annotation["attribute_tokens"]):
                attribute = nusc.get("attribute", annotation["attribute_tokens"][0])[
                    "name"
                ]
                if "parked" in attribute:
                    continue
            else:
                continue

            skip_bool, new_category, ego_category = node_selector_fun(
                category, attribute, env
            )
            if skip_bool:
                continue

            data_point = pd.Series(
                {
                    "frame_id": frame_id,
                    "scene_id": scene_id,
                    "type": new_category,
                    "node_id": annotation["instance_token"],
                    "robot": False,
                    "x": annotation["translation"][0],
                    "y": annotation["translation"][1],
                    "z": annotation["translation"][2],
                    "length": annotation["size"][0],
                    "width": annotation["size"][1],
                    "height": annotation["size"][2],
                    "heading": Quaternion(annotation["rotation"]).yaw_pitch_roll[0],
                }
            )
            data = data.append(data_point, ignore_index=True)

        sample_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        annotation = nusc.get("ego_pose", sample_data["ego_pose_token"])
        data_point = pd.Series(
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "type": env.NodeType.VEHICLE,
                "node_id": "ego",
                "robot": True,
                "x": annotation["translation"][0],
                "y": annotation["translation"][1],
                "z": annotation["translation"][2],
                "length": 4,
                "width": 1.7,
                "height": 1.5,
                "heading": Quaternion(annotation["rotation"]).yaw_pitch_roll[0],
                "orientation": None,
            }
        )
        data = data.append(data_point, ignore_index=True)

        sample = nusc.get("sample", sample["next"])
        frame_id += 1
    if len(data.index) == 0:
        return None
    data.sort_values("frame_id", inplace=True)
    return data


def ns_map2data_map(nusc_map, scene_boundaries, fraction, layer="drivable_area"):
    """
    convert object of NuscMap class to array which is required for creating an object of DataMap class
    """
    x_min, x_max, y_min, y_max = scene_boundaries
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box = (
        x_min + 0.5 * (x_max - x_min),
        y_min + 0.5 * (y_max - y_min),
        y_size,
        x_size,
    )
    patch_angle = 0  # Default orientation where North is up
    canvas_size = (
        np.round(1 / fraction * y_size).astype(int),
        np.round(1 / fraction * x_size).astype(int),
    )
    layer_names = [
        "lane",
        "road_segment",
        "drivable_area",
        "road_divider",
        "lane_divider",
        "stop_line",
        "ped_crossing",
        "stop_line",
        "ped_crossing",
        "walkway",
    ]
    scene_mask = (
        nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
    ).astype(np.uint8)
    if layer == "drivable_area":
        scene_mask = np.max(scene_mask[:3], axis=0, keepdims=True)
    elif layer == "road_divider":
        scene_mask = scene_mask[3]
    elif layer == "lane_divider":
        scene_mask = scene_mask[4]
    elif layer == "all":
        scene_mask = np.stack(
            (np.max(scene_mask[:3], axis=0), scene_mask[3], scene_mask[4]), axis=0
        )
    return scene_mask
