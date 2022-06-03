"""
This module contains utils for data loading used in trainining
"""
import sys
import dill
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

sys.path.append("../../../Trajectron/trajectron")
from model.dataset import EnvironmentDataset, collate


def shorten_dataset(dataset, num_samples):
    """
    Shorten dataset to the given num_samples.
    If num_samples is None, return the original dataset instead.
    :param dataset:
    :param num_samples:
    :return dataset:
    """
    if num_samples is None:
        random_subset = dataset
    elif num_samples >= len(dataset):
        random_subset = dataset
    elif num_samples < len(dataset):
        idx = torch.randperm(len(dataset))[:num_samples]
        random_subset = Subset(dataset, idx)
    return random_subset


def shuffle_dataset(dataset):
    """
    Returns dataset with it's samples in a random order.

    :param dataset:
    :return shuffled_dataset
    """
    idx = torch.randperm(len(dataset))
    shuffled_dataset = Subset(dataset, idx)
    return shuffled_dataset


def concat_datasets(dataset_list, num_samples_total=None, percentages=None):
    """
    Concatenate all datasets in dataset_list to a single dataset.

    :param dataset_list: list of torch datasets
    :param num_samples_total: The single dataset will be of the length given by num_samples_total
    :param percentages: list of datasets that determine the part of each individual dataset
    :return concat_dataset
    """
    concat_list = []
    if len(dataset_list) == 1:
        return dataset_list[0]

    for i, dataset in enumerate(dataset_list):
        if not None in percentages and num_samples_total is not None:
            num_dataset_samples = int(percentages[i] * num_samples_total)
            assert num_dataset_samples <= len(dataset)
        else:
            num_dataset_samples = len(dataset)
        idx = torch.randperm(len(dataset))[:num_dataset_samples]
        subset = Subset(dataset, idx)
        concat_list.append(subset)
    concat_dataset = ConcatDataset(concat_list)
    return concat_dataset


def prepare_dataset(data_path, hyperparams, args):
    """
    Load data from data_path and return different representations of it for different uses

    :param train_data_path:
    :param hyperparams:
    :param args:
    :return: env, dataset, scenes_sample_probs, scenes
    """
    with open(data_path, "rb") as file:
        env = dill.load(file, encoding="latin1")
    print(f"\n\n\nLoaded data from {data_path}")

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(" ")
        env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if env.robot_type is None and hyperparams["incl_robot_node"]:
        env.robot_type = env.NodeType[0]
        for scene in env.scenes:
            scene.add_robot_from_nodes(env.robot_type)

    scenes = env.scenes
    scenes_sample_probs = (
        env.scenes_freq_mult_prop if args.scene_freq_mult_train else None
    )
    dataset = EnvironmentDataset(
        env,
        hyperparams["state"],
        hyperparams["pred_state"],
        data_dir=data_path,
        scene_freq_mult=hyperparams["scene_freq_mult_train"],
        node_freq_mult=hyperparams["node_freq_mult_train"],
        hyperparams=hyperparams,
        min_history_timesteps=hyperparams["minimum_history_length"],
        min_future_timesteps=hyperparams["prediction_horizon"],
        return_robot=hyperparams["return_robot"],
        min_scene_interaction_density=args.min_scene_interaction_density,
    )
    return env, dataset, scenes_sample_probs, scenes


def prepare_dataloader(datasets, node_type, args, train):
    """
    Create dataloader from datasets

    :param train_datasets:
    :param node_type:
    :param args:
    :param train:
    :return:
    """
    dataloader = {}
    node_type_datasets = []
    for dataset in datasets:
        for node_type_dataset in dataset:
            if len(node_type_dataset) == 0:
                continue
            if not (node_type_dataset.node_type == node_type):
                continue
            node_type_datasets.append(node_type_dataset)

    if len(node_type_datasets) > 1:
        node_type_dataset = concat_datasets(
            dataset_list=node_type_datasets,
            num_samples_total=args.num_train_samples
            if train
            else args.num_eval_samples,
            percentages=args.percentages,
        )
    else:
        node_type_dataset = node_type_datasets[0]

    if args.node_freq_mult_train:
        if "nuScenes" in node_type_dataset.data_dir:
            args.num_train_samples = int(7.55 * args.num_train_samples)
        if "openDD" in node_type_dataset.data_dir:
            args.num_train_samples = int(2.0 * args.num_train_samples)

    print(
        f"before shorten:{len(node_type_dataset)} {'train' if train else 'eval'} samples"
    )
    node_type_dataset = shorten_dataset(
        dataset=node_type_dataset,
        num_samples=args.num_train_samples if train else args.num_eval_samples,
    )
    print(
        f"after shorten:{len(node_type_dataset)} {'train' if train else 'eval'} samples\n\n\n"
    )
    node_type_dataloader = DataLoader(
        node_type_dataset,
        collate_fn=collate,
        pin_memory=not args.device == "cpu",
        batch_size=args.batch_size,
        shuffle=args.shuffle_loader,
        num_workers=args.preprocess_workers,
    )
    dataloader[node_type] = node_type_dataloader
    return dataloader
