"""
This Module splits the datasets according to driving characteristics
"""
import sys
import os
import random
import json
import argparse
import warnings
import dill
import torch
import numpy as np
from torch.serialization import SourceChangeWarning
sys.path.append("../../experiments/Trajectron/data_analysis")
sys.path.append("../../experiments/Trajectron/experiment_utils")
sys.path.append("../converter_function")
from scene_analysis import (
    split_env_by_condition,
    node_stands_on_all_timesteps,
    node_stands_on_some_timestep,
    node_no_acceleration,
    node_acceleration,
    node_negative_acceleration,
    no_turning_angle,
    low_turning_angle,
    mid_turning_angle,
    high_turning_angle,
    node_curvature_change,
    low_distance,
    mid_distance,
    high_distance,
)
from experiment_utils import load_domain_env, load_model
from preprocessing_utils import ensure_dir

warnings.filterwarnings("ignore", category=SourceChangeWarning)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf",
    help="model config notwendig nur für die hyperparamer des Datensatzes.",
    type=str,
    default="../../experiments/Trajectron/models/original_config/robot/config.json",
)

parser.add_argument(
    "--data_dirs",
    help="Liste der Datensätze, die nach ihren Fahrcharakteristiken aufgespalten werden sollen.",
    type=str,
    nargs="*",
    default=[
        "nuScenes_Queenstown_middle",
        "nuScenes_Onenorth_middle",
        "nuScenes_Boston_middle",
        "lyft_level_5_middle",
        "openDD",
    ],
)
parser.add_argument(
    "--processed_path", default="../../datasets/processed/Trajectron_format/"
)


parser.add_argument(
    "--speed", help="Geschwindigkeitsbereich des Datensatzes.", type=str, default="_"
)

parser.add_argument("--override_attention_radius", default=[])
parser.add_argument("--scene_freq_mult_train", default=False)
parser.add_argument("--return_robot", default=False)
parser.add_argument("--min_scene_interaction_density", default=-1)


args = parser.parse_args()


def write_data(args, train, env_true, domain, split_by, category):
    """
    Write environment

    :param args:
    :param train:
    :param env_true:
    :param domain:
    :param split_by:
    :param category:
    """
    output_dir = os.path.join(
        args.processed_path,
        "..",
        "split_by_characteristics",
        split_by,
        category,
        domain + ("_train" if train else ""),
    )
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, os.path.basename(domain) + "_" + "test")
    with open(output_path, "wb") as file:
        dill.dump(env_true, file, protocol=dill.HIGHEST_PROTOCOL)


def main():
    """
    Split all environments into driving characteristics and write the parts into separate files
    """
    for data_dir in args.data_dirs:
        for train in [True, False]:
            if train and not "lyft_level_5" in data_dir:
                continue

            with open(args.conf, "r", encoding="utf-8") as conf_json:
                hyperparams = json.load(conf_json)
            env = load_domain_env(
                args,
                hyperparams,
                data_dir=os.path.join(args.processed_path, data_dir),
                train=train,
                speed=args.speed,
            )

            ###################################
            # BEWEGUNGSART
            ###################################
            split_by = "Bewegungsart"
            env_true, env_false = split_env_by_condition(
                env, and_conditions=[node_stands_on_all_timesteps], bools=[True]
            )
            write_data(args, train, env_true, data_dir, split_by, category="stand")
            write_data(
                args, train, env_false, data_dir, split_by, category="start_or_move"
            )

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[
                    node_stands_on_all_timesteps,
                    node_stands_on_some_timestep,
                ],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="start")

            env_true, env_false = split_env_by_condition(
                env, and_conditions=[node_stands_on_some_timestep], bools=[False]
            )
            write_data(args, train, env_true, data_dir, split_by, category="move")

            ###################################
            # BESCHLEUNIGUNG
            ###################################
            split_by = "Beschleunigung"
            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, node_no_acceleration],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="const")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, node_acceleration],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="acc")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[
                    node_stands_on_some_timestep,
                    node_negative_acceleration,
                ],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="dec")

            #############################################
            # KURSÄNDERUNG
            #############################################
            split_by = "Kursaenderung"
            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, no_turning_angle],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="00_15")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, low_turning_angle],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="15_45")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, mid_turning_angle],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="45_75")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, high_turning_angle],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="75_360")

            #############################################
            # KRÜMMUNGSÄNDERUNG
            #############################################
            split_by = "Lenkrichtung"
            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[
                    node_stands_on_some_timestep,
                    no_turning_angle,
                    node_curvature_change,
                ],
                bools=[False, False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="change")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, node_curvature_change],
                bools=[False, False],
            )
            write_data(args, train, env_true, data_dir, split_by, category="keep")

            #############################################
            # DISTANZ
            #############################################
            split_by = "Distanz"
            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, low_distance],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="short")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, mid_distance],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="mid")

            env_true, env_false = split_env_by_condition(
                env,
                and_conditions=[node_stands_on_some_timestep, high_distance],
                bools=[False, True],
            )
            write_data(args, train, env_true, data_dir, split_by, category="long")


if __name__ == "__main__":
    main()
