import os
import sys
import random
import numpy as np
import torch
from argparse import ArgumentParser
from trainer import trainer_ae, trainer_controllerMem, trainer_IRM
from datetime import datetime
from trainer.load_data import load_data_from_env, dataset_sample_plots


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--withIRM", type=bool, default=True)
    parser.add_argument("--use_robot_trajectories", type=bool, default=False)
    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--percentages", type=float, default=None)
    #parser.add_argument("--percentages", type=float, nargs='*')
    parser.add_argument("--speed", default='middle')
    parser.add_argument("--shuffle_loader", type=bool, default=True)


    # data_loader parameters
    parser.add_argument("--past_len", type=int, default=4, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dt", type=float, default=0.5, help="Sample zeit der beim Training verwendeten daten, daten müssen vorher schon so angepasst sein!")
    parser.add_argument("--dim_clip", type=int, default=175, help="total size of the map window around the vehicle (in pixels)")
    parser.add_argument("--data_dir_train", type=str, help="directory of files used for training. If no file as argument, whole dir will be used!")
    parser.add_argument("--data_dir_eval", type=str, help="directory of files used for eval. If no file as argument, whole dir will be used!")
    parser.add_argument("--dataset_file_train", type=str, default=None, help="dataset file with training samples")
    parser.add_argument("--dataset_file_eval", type=str, default=None, help="dataset file with evaluation samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--preprocess_workers", default=0)


    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    parser.add_argument("--preds", type=int, default=10)
    parser.add_argument("--model_ae", type=str, default='pretrained_models/model_AE/model_ae')

    parser.add_argument("--model", type=str, default='pretrained_models/model_controller/model_controller')
    parser.add_argument("--saved_memory", default=True)
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--saveImages", default=False, help="plot qualitative examples in tensorboard")

    return parser.parse_args()

def main(config, dataset, time):
    print(f'Training started at: {time}')
    print(f'Dataset: {dataset}')

    print(f'DATASET PREPROCESSING: train data')
    train_dataset, train_dataloader = load_data_from_env(config, data_dir=config.data_dir_train, data_file=config.dataset_file_train, train=True)
    dataset_sample_plots(train_dataset, config.dim_clip)

    print(f'DATASET PREPROCESSING: eval data')
    dataset_and_loader_eval = load_data_from_env(config, data_dir=config.data_dir_eval, data_file=config.dataset_file_eval, train=False)
    dataset_and_loader = (*(train_dataset, train_dataloader), *dataset_and_loader_eval)
    print('dataset created')


    print('INIT AUTOENCODER TRAINER')
    t_ae = trainer_ae.Trainer(config, dataset, time, dataset_and_loader)
    print('START TRAINING AUTOENCODER')
    t_ae.fit()

    # print('INIT CONTROLLER TRAINER')
    # t_c = trainer_controllerMem.Trainer(config, dataset, time, dataset_and_loader)
    # print('START TRAINING CONTROLLER')
    # t_c.fit()
    #
    # print('INIT ITERATIVE REFINEMENT MODULE TRAINER')
    # t_IRM = trainer_IRM.Trainer(config, dataset, time, dataset_and_loader)
    # print('START TRAINING ITERATIVE REFINEMENT MODULE')
    # t_IRM.fit()


if __name__ == '__main__':
#python train_all_steps.py --data_dir_train Kitti_middle --data_dir_eval Kitti_middle --use_robot_trajectories True
#python train_all_steps.py --data_dir_train lyft_middle --data_dir_eval lyft_middle
#python train_all_steps.py --data_dir_train openDD_full --data_dir_eval openDD_full --num_train_samples 11000
#python train_all_steps.py --data_dir_train boston_middle --data_dir_eval boston_middle --num_train_samples 11000
#python train_all_steps.py --data_dir_train singapore_hv_qt_middle --data_dir_eval singapore_hv_qt_middle --num_train_samples 11000
#python train_all_steps.py --data_dir_train singapore_on_middle --data_dir_eval singapore_on_middle
#python train_all_steps.py --data_dir_train boston_singapore_fast --data_dir_eval boston_singapore_fast --speed "fast"
#python train_all_steps.py --data_dir_train openDD_boston_middle --data_dir_eval openDD_boston_middle --percentages 0.5 0.5 --num_train_samples 22000

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = parse_config()
    time = str(datetime.now())[:16]
    if config.dataset_file_train is None:
        dataset = config.data_dir_train
    else:
        dataset = os.path.basename(config.dataset_file_train).split('.')[0]
    main(config, dataset, time)
