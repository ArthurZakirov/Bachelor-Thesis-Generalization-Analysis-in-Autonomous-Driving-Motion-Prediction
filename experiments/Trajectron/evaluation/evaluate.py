"""
This Module performs the evaluation of Trajectron+++
"""
import sys
import os
import json
import random
import warnings
from collections import defaultdict
import argparse
import torch
import numpy as np
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)
sys.path.append("../../../Trajectron/trajectron")
sys.path.append("../../../converter/converter_functions")
sys.path.append("../train")
sys.path.append("../../Trajectron/train")
sys.path.append("../experiment_utils")
sys.path.append("../../Trajectron/experiment_utils")

from experiment_utils import load_domain_env, load_model
from model.trajectron import Trajectron, trajectron_const_a
from preprocessing_utils import ensure_dir
from data2env_functions import init_env
from evaluation_utils import (
    split_by_standing_status,
    split_by_driving_characteristics,
    split_by_speed_zone,
    split_by_country_or_road_class,
    evaluate_domain,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


parser = argparse.ArgumentParser()

################################################
# MODEL
################################################
parser.add_argument(
    "--models",
    help='List of model directories inside "experiments/Trajectron/models"',
    type=str,
    nargs="*",
    default=[],
)

parser.add_argument(
    "--checkpoints",
    help="List of checkpoints, for each model. "
    "The Checkpoint ist the epoch, at which the Training stops."
    'If "None", last epoch will be selected.',
    nargs="*",
    type=int,
    default=[None, None, None, None, None, None, None, None],
)

parser.add_argument(
    "--physics_based",
    help="use physics based prediction with constant acc. instead of T++",
    action="store_true",
)

###########################################################
# EVAUATION DATA
###########################################################

parser.add_argument(
    "--data_dirs",
    help="List of data_dirs, that the model should be evaluated on.",
    type=str,
    nargs="*",
)

parser.add_argument(
    "--split_criterium",
    help="Data split criteria: "
    "'split_by_speed_zone', "
    "'split_by_country_or_road_class', "
    "'split_by_driving_characteristics', "
    "'split_by_standing_status'",
    type=str,
    default="split_by_standing_status",
)

parser.add_argument(
    "--speed", help="Speep Zone of Evaluation Datasets.", type=str, default="_"
)

parser.add_argument(
    "--node_type", help="node type to evaluate", type=str, default="VEHICLE"
)

parser.add_argument(
    "--split",
    help="Use 'training' / 'test' / 'val' data for Evalution",
    type=str,
    default="val",
)

parser.add_argument(
    "--num_eval_scenes",
    help="Number of scenes used for Evaluation. If None, all scenes are uses.",
    type=int,
    default=None,
)
################################################
# DATA DESIGN
################################################
parser.add_argument(
    "--ph_list",
    nargs="*",
    help="List of prediction horizons for evaluation (in timesteps, not seconds)",
    type=int,
    default=[12],
)

parser.add_argument(
    "--k",
    nargs="*",
    help="List of number of Modes for Multimodality",
    type=int,
    default=[5],
)

parser.add_argument(
    "--max_hl",
    help="max history length only used for evaluation (in timesteps, not seconds)",
    type=int,
    default=4,
)

parser.add_argument(
    "--return_robot",
    help="If model does not use robot feature, "
    "wether robot trajectories should be part of the evaluation set.",
    action="store_true",
)

################################################
# ALLGEMEINES
################################################
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--experiment_tag", help="path to output csv file", type=str)

################################################
# ALLGEMEINES
################################################
# Irrelevant für die Evaluation. Müssen aber mit übergeben werden.
parser.add_argument("--override_attention_radius", default=[])
parser.add_argument("--scene_freq_mult_train", default=False)
parser.add_argument("--min_scene_interaction_density", type=float, default=-1)

args = parser.parse_args()


def evaluate_all_models_on_same_dataset(args, split_criterium):
    """
    perform the evaluation on the same dataset for each model

    :param split_criterium: rule by which to split data
    """
    results = defaultdict(dict)
    for i, model in enumerate(args.models):
        eval_stg, hyperparams = load_model(model, epoch=args.checkpoints[i])
        eval_stg.hyperparams["return_robot"] = args.return_robot
        eval_stg.max_ht = args.max_hl
        eval_stg.set_environment(init_env())
        for data_dir in args.data_dirs:
            data_dir = split_criterium(data_dir)
            env = load_domain_env(args, hyperparams, data_dir, speed=args.speed)
            eval_dict_raw = evaluate_domain(eval_stg, env.scenes, args, env)

            eval_dict = defaultdict(dict)
            for metric in ["ADE", "FDE", "MR", "tolerance_rate"]:
                for pred_horizon in args.ph_list:
                    eval_dict[metric][pred_horizon] = {}

            for metric, metric_dict in eval_dict_raw.items():
                for pred_horizon, ph_dict in metric_dict.items():
                    for k in ph_dict.keys():
                        value = eval_dict_raw[metric][pred_horizon][k]

                        eval_dict[metric][pred_horizon][k] = {
                            "AB": str(round(value * 100, 2)) + " %"
                            if (metric in ["MR", "tolerance_rate"])
                            else str(round(value, 2)) + " m",
                        }
            results[model][data_dir] = eval_dict

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf8") as file:
        json.dump({"model": "Trajectron++", "results": dict(results)}, file)
    with open(os.path.join(output_dir, "details.txt"), "w", encoding="utf8") as file:
        file.write(f"\ndata_dirs: {str(args.data_dirs)}")
        file.write(f"\nmodels: {str(args.models)}")


def evaluate_all_models_on_different_dataset(args, split_criterium, split_by):
    """
    evaluate all models on different datasets

    :param split_criterium: rule by which to split data
    :param split_by: apply the split_criterium depending on "model" or "data"
    :return:
    """
    results = defaultdict(dict)
    for i, model in enumerate(args.models):
        eval_stg, hyperparams = load_model(model, epoch=args.checkpoints[i])
        eval_stg.hyperparams["return_robot"] = args.return_robot
        eval_stg.max_ht = args.max_hl
        eval_stg.set_environment(init_env())

        if split_by == "data":
            data_dirs = split_criterium(args.data_dirs[i])
        elif split_by == "model":
            data_dirs = split_criterium(args.models[i])

        for data_dir in data_dirs:
            env = load_domain_env(args, hyperparams, data_dir, speed=args.speed)
            eval_dict_raw = evaluate_domain(eval_stg, env.scenes, args, env)

            eval_dict = defaultdict(dict)
            for metric in ["ADE", "FDE", "MR", "tolerance_rate"]:
                for pred_horizon in args.ph_list:
                    eval_dict[metric][pred_horizon] = {}

            for metric, metric_dict in eval_dict_raw.items():
                for pred_horizon, ph_dict in metric_dict.items():
                    for k in ph_dict.keys():
                        value = eval_dict_raw[metric][pred_horizon][k]

                        eval_dict[metric][pred_horizon][k] = {
                            "AB": str(round(value * 100, 2)) + " %"
                            if (metric in ["MR", "tolerance_rate"])
                            else str(round(value, 2)) + " m",
                        }
            results[model][data_dir] = eval_dict

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf8") as file:
        json.dump({"model": "Trajectron++", "results": dict(results)}, file)
    with open(os.path.join(output_dir, "details.txt"), "w", encoding="utf8") as file:
        file.write(f"\ndata_dirs: {str(args.data_dirs)}")
        file.write(f"\nmodels: {str(args.models)}")


if __name__ == "__main__":
    if args.physics_based:
        Trajectron.predict = trajectron_const_a

    output_dir = os.path.join("../results", args.experiment_tag)
    ensure_dir(output_dir)

    if args.split_criterium == "split_by_speed_zone":
        evaluate_all_models_on_different_dataset(
            args, split_criterium=split_by_speed_zone, split_by="data"
        )

    elif args.split_criterium == "split_by_country_or_road_class":
        evaluate_all_models_on_different_dataset(
            args, split_criterium=split_by_country_or_road_class, split_by="model"
        )

    elif args.split_criterium == "split_by_driving_characteristics":
        evaluate_all_models_on_different_dataset(
            args, split_criterium=split_by_driving_characteristics, split_by="data"
        )

    elif args.split_criterium == "split_by_standing_status":
        evaluate_all_models_on_same_dataset(
            args, split_criterium=split_by_standing_status
        )
