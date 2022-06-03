"""
This Module loads nuScenes raw data and converts it to the "env" format
"""
import sys
import os
import argparse
import dill
from tqdm import tqdm
import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.append("../converter_functions")
sys.path.append("../converter_maps")
sys.path.append("../../Trajectron/trajectron")
sys.path.append("../../experiments/Trajectron/data_analysis")
from environment import Scene
from nuScenes2data_functions import (
    ns_scene2data,
    nuScenes_category2env_category,
    split_scene_names,
    ns_map2data_map,
)
from data2env_functions import (
    augment,
    augment_scene,
    shift_data,
    extract_nodes_from_data,
    data_map2env_map,
    init_env,
)
from preprocessing_utils import ensure_dir
from scene_analysis import scene_velocity_stats

SCENE_BLACKLIST = [499, 515, 517]
AUGMENT_ANGLE_STEPS = 15
CURV_0_2 = 0
CURV_0_1 = 0
TOTAL_CURV = 0

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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_data_dir",
    type=str,
    default="../../datasets/raw/nuScenes",
    help="dir that contains [v1.0 / maps/ samples/ sweeps]",
)
parser.add_argument(
    "--mini",
    action="store_true",
    help='Bei Auswahl von "mini" wird die mini-Version von nuScenes-verwendet.',
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="../../datasets/processed/Trajectron_format/nuScenes_road_classes",
    help="directory where to save processes data",
)
parser.add_argument(
    "--val_split",
    type=int,
    default=0.10,
    help="Split data into train=1-2*split, val=split, test=split",
)

parser.add_argument(
    "--middle_speed_lower_bound",
    type=float,
    default=30.0,
    help="Untere Grenze des mittleren Geschwindigkeitsbereichs [km/h]",
)
parser.add_argument(
    "--middle_speed_upper_bound",
    type=float,
    default=65.0,
    help="Obere Grenze des mittleren Geschwindigkeitsbereichs [km/h]",
)
parser.add_argument(
    "--dt", type=float, default=0.5, help="Sample Abtastschrittweite [s]"
)
parser.add_argument(
    "--scene_buffer",
    type=int,
    default=90,
    help="Extra Platz am Rand des Kartenausschnitts der Szene [pixel]",
)
parser.add_argument(
    "--fraction", type=float, default=1 / 3, help="Rastergröße der Karte [m]"
)
parser.add_argument(
    "--layer",
    type=str,
    default="road_divider",
    help="Semantische Kartenschicht. Auswahl zwischen: "
    "'drivable_area', 'road_divider', 'lane_divider', 'all'",
)
args = parser.parse_args()


def process_ns_scene(ns_scene, scene_id, nusc, args):
    """
    process object of the nuScenes-Scene class to object of env-Scene class

    :param ns_scene: nuScenes-Scene class object
    :param scene_id: int number of scene
    :param nusc:
    :param args:
    :return: object of env-Scene class
    """
    env = init_env()
    data = ns_scene2data(nusc, ns_scene, scene_id, env, nuScenes_category2env_category)
    data, scene_boundaries = shift_data(data, args.scene_buffer)

    nusc_map = NuScenesMap(
        dataroot=args.raw_data_dir,
        map_name=nusc.get("log", ns_scene["log_token"])["location"],
    )
    scene_mask = ns_map2data_map(nusc_map, scene_boundaries, args.fraction, args.layer)
    scene = Scene(
        timesteps=data["frame_id"].max() + 1,  # 40
        dt=args.dt,
        name=str(data.loc[0, "scene_id"]),
        aug_func=augment,
    )
    scene.map = data_map2env_map(scene_mask, args.fraction)
    scene.nodes = extract_nodes_from_data(data, scene, env)
    return scene


def main():
    """
    loads nuScenes raw data and converts it to the "env" format
    """
    ensure_dir(args.output_dir)
    version = "v1.0-mini" if args.mini else "v1.0"
    nusc = NuScenes(version=version, dataroot=args.raw_data_dir, verbose=True)
    ns_scene_names = split_scene_names(version, args.val_split)
    locations = [
        "boston-seaport",
        "singapore-onenorth",
        "singapore-queenstown",
        "singapore-hollandvillage",
    ]

    for data_class in ["train", "test", "val"]:
        print(f"\nProcess {len(ns_scene_names[data_class])} {data_class}-scenes.")
        data_class_locs = {loc: [] for loc in locations}
        for loc in locations:
            env_dict = {speed: init_env() for speed in ["slow", "middle", "fast"]}
            scenes_dict = {speed: [] for speed in ["slow", "middle", "fast"]}

            for ns_scene_name in ns_scene_names[data_class]:
                ns_scene = nusc.get(
                    "scene", nusc.field2token("scene", "name", ns_scene_name)[0]
                )
                scene_loc = nusc.get("log", ns_scene["log_token"])["location"]
                if scene_loc == loc:
                    data_class_locs[loc].append(ns_scene)
            print(f"\n...in {loc}.")
            for ns_scene in tqdm(data_class_locs[loc]):
                scene_id = int(ns_scene["name"].replace("scene-", ""))
                if scene_id in SCENE_BLACKLIST:  # Some scenes have bad localization
                    continue
                scene = process_ns_scene(ns_scene, scene_id, nusc, args)
                if scene is not None:
                    if data_class == "train":
                        scene.augmented = []
                        angles = np.arange(0, 360, AUGMENT_ANGLE_STEPS)
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    scene_velocities = scene_velocity_stats(scene)
                    if scene_velocities["max"] < args.middle_speed_lower_bound:
                        scenes_dict["slow"].append(scene)
                    elif scene_velocities["max"] > args.middle_speed_upper_bound:
                        scenes_dict["fast"].append(scene)
                    else:
                        scenes_dict["middle"].append(scene)

            for speed in ["slow", "middle", "fast"]:
                env_dict[speed].scenes = scenes_dict[speed]
                if len(scenes_dict[speed]) > 0:
                    mini_string = ""
                    if args.mini:
                        mini_string = "_mini"
                    data_dict_path = os.path.join(
                        args.output_dir,
                        "nuScenes_"
                        + loc
                        + "_"
                        + speed
                        + "_"
                        + data_class
                        + mini_string
                        + "_"
                        + ".pkl",
                    )
                    with open(data_dict_path, "wb") as file:
                        dill.dump(env_dict[speed], file, protocol=dill.HIGHEST_PROTOCOL)
            del env_dict
            del scenes_dict

            global TOTAL_CURV
            global CURV_0_2
            global CURV_0_1
            TOTAL_CURV = 0
            CURV_0_1 = 0
            CURV_0_2 = 0
    print("finished nuScenes2env!")

    locations = ["Onenorth", "Queenstown", "Boston"]
    speeds = ["slow", "middle", "fast"]
    dirs = [
        "nuScenes" + "_" + loc + "_" + speed for loc in locations for speed in speeds
    ]

    processed_data_dir = os.path.dirname(args.output_dir)

    for directory in dirs:
        ensure_dir(os.path.join(processed_data_dir, directory))

    for file in os.listdir(args.output_dir):
        source_path = os.path.join(args.output_dir, file)

        if "hollandvillage" in file or "queenstown" in file:
            loc = "Queenstown"
        if "boston" in file:
            loc = "Boston"
        if "onenorth" in file:
            loc = "Onenorth"

        for speed in speeds:
            if speed in file:
                if speed != "middle":
                    file = file.replace("train", "test_extra")
                target_path = os.path.join(
                    processed_data_dir, "nuScenes" + "_" + loc + "_" + speed, file
                )
        os.replace(source_path, target_path)


if __name__ == "__main__":
    main()
