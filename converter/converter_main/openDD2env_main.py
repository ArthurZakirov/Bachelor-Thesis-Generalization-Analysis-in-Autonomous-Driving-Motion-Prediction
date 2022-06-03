"""
This Module loads openDD raw data and converts it to the "env" format
"""
import sys
import argparse
sys.path.append("../converter_functions")
sys.path.append("../converter_maps")
from data2env_functions import data2env
from openDD2data_functions import openDD2data
from preprocessing_utils import ensure_dir


parser = argparse.ArgumentParser()

####################################
# Allgemeines
####################################
parser.add_argument(
    "--raw_data_dir",
    type=str,
    default="../../datasets/raw/openDD",
    help="dir that contains [rdb1, ..., rdb7]",
)

parser.add_argument(
    "--mini",
    action="store_true",
    help='Bei Auswahl von "mini" wird die mini-Version von openDD verwendet.',
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="../../datasets/processed/Trajectron_format/openDD",
    help="directory where to save processes data",
)

####################################
# Parameter für Trajektorien
####################################
parser.add_argument(
    "--scene_time", type=float, default=20.0, help="Zeitdauer einzelner Szenen [s]"
)

parser.add_argument(
    "--dt", type=float, default=0.5, help="Sample Abtastschrittweite [s]"
)

parser.add_argument(
    "--node_types_str",
    type=str,
    nargs="+",
    default=[
        "Car",
        "Pedestrian",
        "Medium Vehicle",
        "Trailer",
        "Bus",
        "Heavy Vehicle",
    ],
    help="Verwendete Node Klassen.",
)

parser.add_argument(
    "--max_scene_overlap",
    type=float,
    default=2.0,
    help="How many seconds can the scenes overlap. "
    "Sometimes that is necessary, "
    "because we don't want to miss scenes due to limited robot nodes.",
)

parser.add_argument(
    "--crit_norm_vehicle",
    type=float,
    default=1.0,
    help='Minimum travelled distance [m] of a vehicle to be not considered "parked"',
)

parser.add_argument(
    "--min_node_time",
    type=float,
    default=2.0,
    help="Minimum Node apppearance duration [s] in the clip.",
)

####################################
# Parameter für Karte
####################################

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

parser.add_argument(
    "--lane_radius", type=int, default=4, help="Halbe Breite der Fahrspur [m]"
)

args = parser.parse_args()


def main():
    """
    loads openDD raw data and converts it to the "env" format
    """
    ensure_dir(args.output_dir)
    traj_dict, map_dict = openDD2data(args)
    data2env(args.output_dir, traj_dict, map_dict)
    print("finished openDD2env")


if __name__ == "__main__":
    main()
