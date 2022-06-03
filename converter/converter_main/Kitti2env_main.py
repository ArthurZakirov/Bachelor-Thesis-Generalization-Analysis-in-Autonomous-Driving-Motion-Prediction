"""
This Module loads Kitti raw data and converts it to the "env" format
"""
import sys
import os
import argparse
sys.path.append("../converter_functions")
sys.path.append("../converter_maps")
from Kitti2data_functions import Kitti2data
from data2env_functions import data2env
from preprocessing_utils import ensure_dir


parser = argparse.ArgumentParser()

################################
# Allgemeines
################################
parser.add_argument(
    "--raw_data_dir",
    type=str,
    default="../../datasets/raw/Kitti",
    help="dir that contains [trajectories / maps]",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="../../datasets/processed/Trajectron_format/Kitti_road_classes",
    help="directory where to save processes data",
)

##############################
# Trajectory Parameter
##############################
parser.add_argument(
    "--dt", type=float, default=0.5, help="Sample Abtastschrittweite [s]"
)

parser.add_argument(
    "--scene_time", type=float, default=8.0, help="Zeitdauer einzelner Szenen [s]"
)

parser.add_argument(
    "--min_node_time",
    type=float,
    default=2.0,
    help="Minimum Node apppearance duration [s] in the clip.",
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
    load Kitti raw data and converts it to the "env" format
    """
    for speed in ["middle", "fast"]:
        traj_dict, map_dict = Kitti2data(args, speed)
        data2env(args.output_dir, traj_dict, map_dict, speed)
    print("\n\n\n\nKitti2env finished!")

    speeds = ["slow", "middle", "fast"]
    kitti_tag = "KITTI"
    dirs = [kitti_tag + "_" + speed for speed in speeds]

    processed_data_dir = os.path.dirname(args.output_dir)

    for directory in dirs:
        ensure_dir(os.path.join(processed_data_dir, directory))

    for file in os.listdir(args.output_dir):
        source_path = os.path.join(args.output_dir, file)
        for speed in speeds:
            if speed in file:
                target_path = os.path.join(
                    processed_data_dir, kitti_tag + "_" + speed, file
                )
        os.replace(source_path, target_path)


if __name__ == "__main__":
    main()
