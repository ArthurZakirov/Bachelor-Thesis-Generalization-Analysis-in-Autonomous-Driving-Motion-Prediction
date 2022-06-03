"""
This Module Evaluates MANTRA
"""
import sys
import os
import json
from collections import defaultdict
import random
from argparse import ArgumentParser
import numpy as np
import torch
sys.path.append("../train/trainer")
sys.path.append("../../Trajectron/evaluation")
sys.path.append("../../../converter/converter_functions")
sys.path.append("../experiment_utils")
from evaluation_utils import (
    split_by_standing_status,
    split_by_driving_characteristics,
    split_by_speed_zone,
    split_by_country_or_road_class,
)
from preprocessing_utils import ensure_dir
from experiment_utils import load_model, load_data_from_env
from evaluation_functions import evaluate_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    """
    Parse Arguments
    """
    parser = ArgumentParser()

    #####################################################
    # EXPERIMENT DESIGN
    #####################################################
    parser.add_argument(
        "--models", help="Models used in Evaluation", type=str, nargs="*"
    )

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

    parser.add_argument("--experiment_tag", type=str)

    parser.add_argument(
        "--percentages",
        help="If multiple Datasets:"
        "Ratio in which they are mixed. Name Datasets in alphabet order)",
        type=float,
        nargs="*",
        default=[None],
    )

    #####################################################
    # Model Design
    #####################################################
    parser.add_argument(
        "--preds",
        help="Anzahl der Maximalen Modi für Multimodale Prädiktion.",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--write",
        help="Ob Online Writing verwendet werden soll. "
        "Gebe für jedes zu evaluierende Model individuell an.",
        type=lambda x: x == "True",
        nargs="*",
        default=[False, False, False, False, False, False, False],
    )

    parser.add_argument(
        "--use_map",
        help="Ob die das Iterative Refinement Module in der Evaluation verwendet werden soll."
        "Gebe für jedes zu evaluierende Model individuell an.",
        type=lambda x: x == "True",
        nargs="*",
        default=[True, True, True, True, True, True, True],
    )

    #####################################################
    # Data Design
    #####################################################
    parser.add_argument(
        "--past_len",
        help="length of past (in timesteps, not seconds)",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--future_len",
        help="length of future (in timesteps, not seconds)",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--ph_list",
        help="List of prediction horizons in Evaluation (in timesteps, not seconds)",
        type=int,
        default=[12],
    )

    parser.add_argument(
        "--k_list", help="List of number of Modes in Evaluation", type=int, default=[5]
    )

    parser.add_argument(
        "--use_robot_trajectories",
        help="Benutze Robot Trajektorien im Training.",
        action="store_true",
    )

    parser.add_argument(
        "--speed",
        help="Geschwindigkeitsbereich des Trainingsdatensatzes.",
        type=str,
        default="",
    )

    parser.add_argument(
        "--dt",
        help="Sample zeit [s] der beim Training verwendeten daten.",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--dim_clip",
        help="Größe des Kartenausschnitts in Fahrtrichtung [pixel]",
        type=int,
        default=60,
    )

    #####################################################
    # Allgemeines
    #####################################################
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--shuffle_loader", type=bool, default=True)

    parser.add_argument("--preprocess_workers", default=0)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--num_eval_samples", type=int, default=5000)

    #####################################################
    # ZUSATZ FÜR VISUALISIERUNG
    #####################################################
    # wird nicht für Evaluation benötigt
    parser.add_argument("--scene_name", type=str, default=None)

    parser.add_argument("--node_id", type=str, default=None)

    return parser.parse_args()


def evaluate_all_models_on_same_dataset(split_criterium):
    """
    perform the evaluation on the same dataset for each model

    :param split_criterium: rule by which to split data
    """
    results = defaultdict(dict)
    datasets_and_loaders = [
        load_data_from_env(args, data_dir=split_criterium(data_dir), train=False)
        for data_dir in args.data_dirs
    ]
    model_tags = []
    for i, model in enumerate(args.models):
        model_tag = model
        if not args.use_map[i]:
            model_tag = model_tag.replace("MANTRA", "MANTRA_no_map")
        elif args.write[i]:
            model_tag = model_tag.replace("MANTRA", "MANTRA_online_writing")

        model_tags.append(model_tag)

        for j, (_, loader) in enumerate(datasets_and_loaders):
            mem_n2n = load_model("training_IRM", model)
            mem_n2n.min_tolerance_rate = 1.0
            if loader is None:
                continue

            eval_dict_raw = evaluate_model(
                args,
                loader,
                mem_n2n,
                device=args.device,
                use_map=args.use_map[i],
                write=args.write[i],
            )
            eval_dict = defaultdict(dict)
            for metric in ["ADE", "FDE", "MR"]:
                for pred_horizon in args.ph_list:
                    eval_dict[metric][pred_horizon] = {}

            for metric, metric_dict in eval_dict_raw.items():
                for pred_horizon, ph_dict in metric_dict.items():
                    for k in ph_dict.keys():
                        value = ph_dict[k]
                        eval_dict[metric][pred_horizon][k] = {
                            "AB": str(round(value * 100, 2)) + " %"
                            if metric == "MR"
                            else str(round(value, 2)) + " m",
                        }
            results[model_tag][args.data_dirs[j]] = eval_dict

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf8") as file:
        json.dump({"model": "MANTRA", "results": dict(results)}, file)
    with open(os.path.join(output_dir, "details.txt"), "w", encoding="utf8") as file:
        file.write(f"\ndata_dirs: {str(args.data_dirs)}")
        file.write(f"\nmodels: {str(model_tags)}")


#
def evaluate_all_models_on_different_dataset(split_criterium, split_by):
    """
    evaluate all models on different datasets

    :param split_criterium: rule by which to split data
    :param split_by: apply the split_criterium depending on "model" or "data"
    :return:
    """
    results = defaultdict(dict)

    for i, model in enumerate(args.models):
        if split_by == "data":
            data_dirs = split_criterium(args.data_dirs[i])
        elif split_by == "model":
            data_dirs = split_criterium(args.models[i])

        for j, data_dir in enumerate(data_dirs):
            mem_n2n = load_model("training_IRM", model)
            _, loader = load_data_from_env(args, data_dir=data_dir, train=False)
            if loader is None:
                continue
            eval_dict_raw = evaluate_model(
                args,
                loader,
                mem_n2n,
                device="cpu",
                use_map=args.use_map[i],
                write=args.write[i],
            )

            eval_dict = defaultdict(dict)
            for metric in ["ADE", "FDE", "MR"]:
                for pred_horizon in args.ph_list:
                    eval_dict[metric][pred_horizon] = {}

            for metric, metric_dict in eval_dict_raw.items():
                for pred_horizon, ph_dict in metric_dict.items():
                    for k in ph_dict.keys():
                        value = ph_dict[k]
                        eval_dict[metric][pred_horizon][k] = {
                            "AB": str(round(value * 100, 2)) + " %"
                            if metric == "MR"
                            else str(round(value, 2)) + " m",
                        }
            results[model][data_dirs[j]] = eval_dict

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf8") as file:
        json.dump({"model": "MANTRA", "results": dict(results)}, file)
    with open(os.path.join(output_dir, "details.txt"), "w", encoding="utf8") as file:
        file.write(f"\ndata_dirs: {str(args.data_dirs)}")
        file.write(f"\nmodels: {str(args.models)}")


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join("../results", args.experiment_tag)
    ensure_dir(output_dir)

    if args.split_criterium == "split_by_speed_zone":
        evaluate_all_models_on_different_dataset(
            split_criterium=split_by_speed_zone, split_by="data"
        )

    elif args.split_criterium == "split_by_country_or_road_class":
        evaluate_all_models_on_different_dataset(
            split_criterium=split_by_country_or_road_class, split_by="model"
        )

    elif args.split_criterium == "split_by_driving_characteristics":
        evaluate_all_models_on_different_dataset(
            split_criterium=split_by_driving_characteristics, split_by="data"
        )

    elif args.split_criterium == "split_by_standing_status":
        evaluate_all_models_on_same_dataset(split_by_standing_status)
