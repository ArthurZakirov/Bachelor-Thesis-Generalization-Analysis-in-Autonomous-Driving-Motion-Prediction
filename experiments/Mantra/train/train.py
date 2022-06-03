"""
This Module trains MANTRA
"""
import sys
import random
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import torch
from trainer.trainer_ae import Trainer
from trainer.trainer_IRM import TrainerIRM

sys.path.append("../experiment_utils")
from experiment_utils import load_data_from_env

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


parser = ArgumentParser()

#####################################################
# EXPERIMENT DESIGN
#####################################################
parser.add_argument(
    "--data_dir",
    help="directory of files used for training. If no file as argument, whole dir will be used!",
    type=str,
)

parser.add_argument(
    "--experiment_tag",
    help="Choose between: 'standard', 'Verbesserung/large_scale', 'Verbesserung/domain_mix'",
    type=str,
)

#####################################################
# Model Design
#####################################################
parser.add_argument(
    "--time",
    help="Startzeitpunkt des Trainings des zu ladenden Modells.",
    type=str,
    default=None,
)

parser.add_argument(
    "--training_step",
    help="Trainingsschritt im 2 Stufigen Trainingsverfahren: 'ae' / 'IRM_decoder' ",
    type=str,
    default="ae",
)

parser.add_argument(
    "--withIRM",
    help="Ob im letzten Trainingsschritt nur der decoder, "
    "oder auch das IRM trainiert werden soll.",
    type=bool,
    default=True,
)

parser.add_argument(
    "--preds",
    help="Anzahl der Maximalen Modi für Mulitmodale Prädiktion.",
    type=int,
    default=10,
)

parser.add_argument(
    "--dim_embedding_key",
    help="Dimensionalität des Feature Vektors der Vergangenheit.",
    type=int,
    default=48,
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
# Training Design
#####################################################
parser.add_argument("--learning_rate", type=float, default=0.0002)

parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("--max_epochs", type=int, default=1000)

parser.add_argument("--num_train_samples", type=int, default=10000)

parser.add_argument(
    "--percentages",
    help="If multiple Datasets used for Training, "
    "give a list with the ratio of the sets in alphabetic order."
    "If there is either only one Dataset, "
    "or the ratio is determined by original Dataset sizes, leave at default.",
    type=float,
    nargs="*",
    default=[None],
)

#####################################################
# Training Configuration
#####################################################
parser.add_argument("--device", type=str, default="cpu")

parser.add_argument("--shuffle_loader", type=bool, default=True)

parser.add_argument("--preprocess_workers", default=0)

parser.add_argument(
    "--resume",
    help="Lade ein gegebenes Modell und setzte das Training fort.",
    action="store_true",
)

parser.add_argument(
    "--saved_memory",
    help="Benutze einen vorgefertigten Speicher beim IRM Training, "
    "anstatt Memory writing.",
    type=bool,
    default=False,
)

#####################################################
# Evaluation
#####################################################
parser.add_argument("--num_eval_samples", type=int, default=1000)

parser.add_argument("--eval_every", type=int, default=1)

#####################################################
# ZUSATZ FÜR VISUALISIERUNG
#####################################################
# wird nicht für Training benötigt
parser.add_argument("--scene_name", type=str, default=None)

parser.add_argument("--node_id", type=str, default=None)

args = parser.parse_args()


def main():
    """
    main function
    """

    train_dataset, train_dataloader = load_data_from_env(
        args, data_dir=args.data_dir, train=True
    )

    eval_dataset, eval_dataloader = load_data_from_env(
        args, data_dir=args.data_dir, train=False
    )

    dataset_and_loader = (
        train_dataset,
        train_dataloader,
        eval_dataset,
        eval_dataloader,
    )

    args.dataset = args.data_dir

    if args.training_step == "ae":
        args.time = str(datetime.now())[:16].replace(":", "-")
        trainer_ae = Trainer(args, dataset_and_loader)
        trainer_ae.fit()

    elif args.training_step == "IRM_decoder":
        trainer_irm = TrainerIRM(args, dataset_and_loader)
        trainer_irm.fit()

    else:
        print(
            'Kein gültiger Trainingsschritt. Setze "--training_step ae" oder "--training_step IRM"!'
        )


if __name__ == "__main__":
    main()
