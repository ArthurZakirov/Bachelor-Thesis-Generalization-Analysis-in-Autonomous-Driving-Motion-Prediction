"""
This Module loads lyft raw data and converts it to the "env" format
"""
import sys
import os
import random
import argparse
import dill
import numpy as np
from tqdm import tqdm
from lyft_dataset_sdk.lyftdataset import LyftDataset

sys.path.append("../converter_functions")
sys.path.append("../converter_maps")
sys.path.append("../../experiments/Trajectron/data_analysis")
from DataMap import DataMap
from data2env_functions import init_env, process_scene, augment_scene
from nuScenes2data_functions import ns_scene2data
from lyft2data_functions import process_lyft_map, lyft_category2env_category
from preprocessing_utils import ensure_dir, interpolate, remove_short_appearing_nodes
from scene_analysis import scene_velocity_stats

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

TOTAL_CURV = 0
CURV_0_1 = 0
CURV_0_2 = 0

parser = argparse.ArgumentParser()

################################
# Allgemeines
################################
parser.add_argument(
    "--raw_data_dir",
    type=str,
    default="../../datasets/raw/lyft",
    help="dir that contains [train_data / maps]",
)

parser.add_argument(
    "--mini",
    action="store_true",
    help='Bei Auswahl von "mini" wird die mini-Version von nuScenes-verwendet.',
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="../../datasets/processed/Trajectron_format/lyft_road_classes",
    help="directory where to save processes data",
)

parser.add_argument(
    "--val_split",
    type=int,
    default=0.10,
    help="Split data into train=(1-split), val=split",
)

#####################################
# Trajektorien Parameter
#####################################
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

##############################
# Map Parameter
##############################
parser.add_argument(
    "--scene_buffer",
    type=int,
    default=90,
    help="Extra Platz am Rand des Kartenausschnitts der Szene [pixel]",
)

parser.add_argument(
    "--map_buffer",
    type=int,
    default=100,
    help="Extra Platz am Rand des Kartenausschnitts der Gesamten Karte [pixel]",
)

parser.add_argument(
    "--fraction", type=float, default=1 / 3, help="Rastergröße der Karte [m]"
)

args = parser.parse_args()


def main():
    """
    load lyft raw data and converts it to the "env" format
    """
    ensure_dir(args.output_dir)

    print("Process Lyft Map..")
    map_mask = process_lyft_map(
        os.path.join(args.raw_data_dir, "maps/map_raster_palo_alto.png"), args
    )
    data_map = DataMap(map_mask, args)
    nusc = LyftDataset(
        data_path=args.raw_data_dir,
        json_path=os.path.join(args.raw_data_dir, "train_data"),
    )
    if args.mini:
        nusc.scene = nusc.scene[:5]
    num_scenes = len(nusc.scene)
    scene_ids = list(range(num_scenes))
    random.shuffle(scene_ids)
    splits_dict = {
        "train": [
            nusc.scene[i] for i in scene_ids[: int(num_scenes * (1 - args.val_split))]
        ],
        "val": [nusc.scene[i] for i in scene_ids[-int(num_scenes * args.val_split) :]],
    }
    for split, split_scenes in splits_dict.items():
        print(f"Process {len(split_scenes)} {split}-scenes")
        env_dict = {speed: init_env() for speed in ["slow", "middle", "fast"]}
        scenes_dict = {speed: [] for speed in ["slow", "middle", "fast"]}
        for scene_id, ns_scene in enumerate(tqdm(split_scenes, desc=f"{split} scene")):
            data = ns_scene2data(
                nusc, ns_scene, str(scene_id), init_env(), lyft_category2env_category
            )
            data["t"] = data["frame_id"].astype(float) * 0.2  # 0.2 is original dt
            data = remove_short_appearing_nodes(
                data, min_node_time=1.0, id_feature="node_id", t_feature="t"
            )
            data = interpolate(
                data,
                dt=args.dt,
                interp_features=["x", "y", "heading"],
                categoric_features=["type", "robot", "scene_id"],
                id_feature="node_id",
                t_feature="t",
            )
            scene = process_scene(init_env(), data, data_map, dt=args.dt)
            num_vehicles = len(
                [str(node.type) for node in scene.nodes if str(node.type) == "VEHICLE"]
            )
            if (scene is not None) and (num_vehicles >= 1):
                scene_velocities = scene_velocity_stats(scene)
                if split == "train":
                    augmented_scenes = []
                    for angle in np.arange(0, 360, 15):
                        augmented_scenes.append(augment_scene(scene, angle))
                    scene.augmented = augmented_scenes

                if scene_velocities["max"] < args.middle_speed_lower_bound:
                    scenes_dict["slow"].append(scene)
                elif scene_velocities["max"] > args.middle_speed_upper_bound:
                    scenes_dict["fast"].append(scene)
                else:
                    scenes_dict["middle"].append(scene)

        for speed in ["slow", "middle", "fast"]:
            env_dict[speed].scenes = scenes_dict[speed]
            if len(scenes_dict[speed]) > 0:
                data_dict_path = os.path.join(
                    args.output_dir, "lyft_" + speed + "_" + split + ".pkl"
                )
                with open(data_dict_path, "wb") as file:
                    dill.dump(env_dict[speed], file, protocol=dill.HIGHEST_PROTOCOL)

            global TOTAL_CURV
            global CURV_0_2
            global CURV_0_1
            TOTAL_CURV = 0
            CURV_0_1 = 0
            CURV_0_2 = 0
    print("\n\n\n\nlyft2env finished!")

    speeds = ["slow", "middle", "fast"]
    lyft_tag = "lyft_level_5"
    dirs = [lyft_tag + "_" + speed for speed in speeds]

    processed_data_dir = os.path.dirname(args.output_dir)

    for directory in dirs:
        ensure_dir(os.path.join(processed_data_dir, directory))

    for file in os.listdir(args.output_dir):
        source_path = os.path.join(args.output_dir, file)
        for speed in speeds:
            if speed in file:
                if speed != "middle":
                    file = file.replace("train", "test_extra")
                target_path = os.path.join(
                    processed_data_dir, lyft_tag + "_" + speed, file
                )
        os.replace(source_path, target_path)


if __name__ == "__main__":
    main()
