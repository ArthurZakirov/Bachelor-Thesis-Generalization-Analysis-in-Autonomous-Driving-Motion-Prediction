"""
This module performs the Training of Trajectron++
"""
import os
import sys
import json
import random
import pathlib
from datetime import datetime
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

sys.path.append("../../../Trajectron/trajectron")
sys.path.append("../../../converter/converter_main")
sys.path.append("../../../converter/converter_functions")
sys.path.append("../../../converter/converter_maps")
sys.path.append("..")
sys.path.append("../experiment_utils")
sys.path.append("../evaluation")
import visualization
import evaluation_utils
from experiment_utils import load_model
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from train_utils import prepare_dataset, prepare_dataloader
from argument_parser import args

torch.autograd.set_detect_anomaly(True)


if not torch.cuda.is_available() or args.device == "cpu":
    args.device = torch.device("cpu")
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = "cuda:0"
        torch.cuda.set_device(args.device)

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = torch.device("cpu")


if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    """
    Train Trajectron++
    :return: None
    """
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print("Config json not found!")
    with open(args.conf, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams["dynamic_edges"] = args.dynamic_edges
    hyperparams["edge_state_combine_method"] = args.edge_state_combine_method
    hyperparams["edge_influence_combine_method"] = args.edge_influence_combine_method
    hyperparams["edge_addition_filter"] = args.edge_addition_filter
    hyperparams["edge_removal_filter"] = args.edge_removal_filter
    hyperparams["batch_size"] = args.batch_size
    hyperparams["k_eval"] = args.k_eval
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node
    hyperparams["node_freq_mult_train"] = args.node_freq_mult_train
    hyperparams["node_freq_mult_eval"] = args.node_freq_mult_eval
    hyperparams["scene_freq_mult_train"] = args.scene_freq_mult_train
    hyperparams["scene_freq_mult_eval"] = args.scene_freq_mult_eval
    hyperparams["scene_freq_mult_viz"] = args.scene_freq_mult_viz
    hyperparams["edge_encoding"] = args.edge_encoding
    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["augment"] = args.augment
    hyperparams["override_attention_radius"] = args.override_attention_radius

    # overwrite hyperparams
    hyperparams["prediction_horizon"] = args.ph
    hyperparams["maximum_history_length"] = args.max_hl
    hyperparams["minimum_history_length"] = args.max_hl
    hyperparams["learning_rate"] = args.lr
    hyperparams["map_encoder"]["VEHICLE"]["map_channels"] = len(args.map_layers)
    hyperparams["map_encoder"]["VEHICLE"]["patch_size"] = args.patch_size

    # add my own hyperparams
    hyperparams["return_robot"] = args.return_robot
    hyperparams["use_layers"] = args.map_layers

    # Load training and evaluation environments and scenes
    data_dir = os.path.join(
        "../../../datasets/processed/Trajectron_format", args.data_dir
    )

    train_scenes_all_datasets = []
    train_datasets = []
    if args.train_data_file is None:
        for data_file in sorted(os.listdir(data_dir)):
            if not "train" in data_file:
                continue
            if not args.speed in data_file:
                continue
            train_data_path = os.path.join(data_dir, data_file)
            (
                train_env,
                train_dataset,
                train_scenes_sample_probs,
                train_scenes,
            ) = prepare_dataset(train_data_path, hyperparams, args)
            train_datasets.append(train_dataset)
            train_scenes_all_datasets.append(train_scenes)
        train_scenes = [
            scene
            for dataset_scenes in train_scenes_all_datasets
            for scene in dataset_scenes
        ]
        train_env.scenes = train_scenes
    else:
        train_data_path = os.path.join(data_dir, args.train_data_file)
        (
            train_env,
            train_dataset,
            train_scenes_sample_probs,
            train_scenes,
        ) = prepare_dataset(train_data_path, hyperparams, args)
        train_datasets.append(train_dataset)
    node_type = train_env.NodeType.VEHICLE
    train_data_loader = prepare_dataloader(train_datasets, node_type, args, train=True)

    eval_scenes_all_datasets = []
    eval_scenes = []
    eval_datasets = []
    if args.eval_every is not None:
        if args.eval_data_file is None:
            for data_file in os.listdir(data_dir):
                if not (
                    ("eval" in data_file)
                    or ("test" in data_file)
                    or ("val" in data_file)
                ):
                    continue
                if not args.speed in data_file:
                    continue
                eval_data_path = os.path.join(data_dir, data_file)
                (
                    eval_env,
                    eval_dataset,
                    _,
                    eval_scenes,
                ) = prepare_dataset(eval_data_path, hyperparams, args)
                eval_datasets.append(eval_dataset)
                eval_scenes_all_datasets.append(eval_scenes)
            eval_scenes = [
                scene
                for dataset_scenes in eval_scenes_all_datasets
                for scene in dataset_scenes
            ]
            eval_env.scenes = eval_scenes
        else:
            eval_data_path = os.path.join(data_dir, args.eval_data_file)
            (
                eval_env,
                eval_dataset,
                _,
                eval_scenes,
            ) = prepare_dataset(eval_data_path, hyperparams, args)
            eval_datasets.append(eval_dataset)
        node_type = eval_env.NodeType.VEHICLE
        eval_data_loader = prepare_dataloader(
            eval_datasets, node_type, args, train=False
        )

    # Offline Calculate Scene Graph
    if hyperparams["offline_scene_graph"] == "yes":
        print("Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(
                train_env.attention_radius,
                hyperparams["edge_addition_filter"],
                hyperparams["edge_removal_filter"],
            )
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(
                eval_env.attention_radius,
                hyperparams["edge_addition_filter"],
                hyperparams["edge_removal_filter"],
            )
            print(f"Created Scene Graph for Evaluation Scene {i}")

    if args.train_data_file is None:
        log_tag = f"{args.model_tag}_{args.data_dir}_{args.num_train_samples}_samples"
    else:
        log_tag = (
            f"{args.model_tag}_{args.train_data_file}_{args.num_train_samples}_samples"
        )

    if not args.debug and args.load_model_dir is None:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(
            args.log_dir,
            args.experiment_tag,
            args.model_tag,
            log_tag + "_" + str(datetime.now())[:16].replace(":", "-"),
        )
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        # Save config to model directory
        with open(os.path.join(model_dir, "config.json"), "w") as conf_json:
            json.dump(hyperparams, conf_json)
        model_registrar = ModelRegistrar(model_dir, args.device)
        log_writer = SummaryWriter(log_dir=model_dir)
        trajectron = Trajectron(model_registrar, hyperparams, log_writer, args.device)
        start_epoch = 0
        curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    else:
        model_dir = os.path.join(args.log_dir, args.load_model_dir)
        trajectron, _, last_checkpoint = load_model(
            model_dir, return_last_checkpoint=True
        )
        log_writer = SummaryWriter(log_dir=model_dir)
        trajectron.log_writer = log_writer
        model_registrar = trajectron.model_registrar
        start_epoch = last_checkpoint + 1
        curr_iter_node_type = {
            node_type: int(
                last_checkpoint * len(train_data_loader[train_env.NodeType.VEHICLE])
            )
            for node_type in train_data_loader.keys()
        }
    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print("Created Training Model.")
    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(
            model_registrar, hyperparams, log_writer, args.eval_device
        )
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print("Created Evaluation Model.")

    optimizer = {}
    lr_scheduler = {}
    for node_type in train_env.NodeType:
        if node_type not in hyperparams["pred_state"]:
            continue
        optimizer[node_type] = optim.Adam(
            [
                {
                    "params": model_registrar.get_all_but_name_match(
                        "map_encoder"
                    ).parameters()
                },
                {
                    "params": model_registrar.get_name_match(
                        "map_encoder"
                    ).parameters(),
                    "lr": 0.0008,
                },
            ],
            lr=hyperparams["learning_rate"],
        )
        # Set Learning Rate
        if hyperparams["learning_rate_style"] == "const":
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=1.0
            )
        elif hyperparams["learning_rate_style"] == "exp":
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=hyperparams["learning_decay_rate"]
            )

        if args.load_model_dir is not None and node_type == train_env.NodeType.VEHICLE:
            if os.path.exists(os.path.join(model_dir, "optimizer")):
                (
                    optimizer_dict,
                    scheduler_dict,
                ) = model_registrar.load_optimizer_and_scheduler_dict()
                optimizer[node_type].load_state_dict(optimizer_dict)
                lr_scheduler[node_type].load_state_dict(scheduler_dict)

    #################################
    #           TRAINING            #
    #################################

    for epoch in range(start_epoch, start_epoch + args.train_epochs):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        for node_type, data_loader in train_data_loader.items():
            if node_type == train_env.NodeType.PEDESTRIAN:
                continue
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80)

            for batch in pbar:
                # Training Loss
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                train_loss = trajectron.train_loss(batch, node_type)
                pbar.set_description(
                    f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}"
                )
                train_loss.backward()

                # Clipping gradients.
                if hyperparams["grad_clip"] is not None:
                    nn.utils.clip_grad_value_(
                        model_registrar.parameters(), hyperparams["grad_clip"]
                    )
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                if args.use_lr_scheduler:
                    lr_scheduler[node_type].step()

                if not args.debug:
                    log_writer.add_scalar(
                        f"{node_type}/train/learning_rate",
                        lr_scheduler[node_type].get_last_lr()[0],
                        curr_iter,
                    )
                    log_writer.add_scalar(
                        f"{node_type}/train/loss", train_loss, curr_iter
                    )

                curr_iter += 1

            curr_iter_node_type[node_type] = curr_iter

        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if (
            args.vis_every is not None
            and not args.debug
            and epoch % args.vis_every == 0
            and epoch > 0
        ):
            max_hl = hyperparams["maximum_history_length"]
            ph = hyperparams["prediction_horizon"]
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron.predict(
                    scene,
                    timestep,
                    ph,
                    min_future_timesteps=ph,
                    z_mode=True,
                    gmm_mode=True,
                    all_z_sep=False,
                    full_dist=False,
                )

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(
                    ax,
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    map=scene.map["VISUALIZATION"] if scene.map is not None else None,
                )
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure("train/prediction", fig, epoch)

                model_registrar.to(args.eval_device)
                # Predict random timestep to plot for eval data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(eval_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = eval_trajectron.predict(
                    scene,
                    timestep,
                    ph,
                    num_samples=20,
                    min_future_timesteps=ph,
                    z_mode=False,
                    full_dist=False,
                )

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(
                    ax,
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    map=scene.map["VISUALIZATION"] if scene.map is not None else None,
                )
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure("eval/prediction", fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions = eval_trajectron.predict(
                    scene,
                    timestep,
                    ph,
                    min_future_timesteps=ph,
                    z_mode=True,
                    gmm_mode=True,
                    all_z_sep=True,
                    full_dist=False,
                )

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(
                    ax,
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    map=scene.map["VISUALIZATION"] if scene.map is not None else None,
                )
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure("eval/prediction_all_z", fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if (
            args.eval_every is not None
            and not args.debug
            and epoch % args.eval_every == 0
            and epoch > 0
        ):
            max_hl = hyperparams["maximum_history_length"]
            ph = hyperparams["prediction_horizon"]
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    if node_type == eval_env.NodeType.PEDESTRIAN:
                        continue
                    eval_loss = []
                    print(
                        f"Starting Evaluation @ epoch {epoch} for node type: {node_type}"
                    )
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(
                            batch, node_type
                        )
                        pbar.set_description(
                            f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}"
                        )
                        eval_loss.append({node_type: {"nll": [eval_loss_node_type]}})
                        del batch

                    evaluation_utils.log_batch_errors(
                        eval_loss, log_writer, f"{node_type}/eval_loss", epoch
                    )

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in tqdm(
                    eval_scenes[: args.num_eval_scenes],
                    desc="Sample Evaluation",
                    ncols=80,
                ):
                    timesteps = np.arange(scene.timesteps)

                    predictions = eval_trajectron.predict(
                        scene,
                        timesteps,
                        ph,
                        num_samples=10,
                        min_future_timesteps=ph,
                        z_mode=False,
                        gmm_mode=True,
                        full_dist=False,
                        all_z_sep=False,
                        min_k=True,
                    )

                    eval_batch_errors.append(
                        evaluation_utils.compute_batch_statistics(
                            predictions,
                            scene.dt,
                            max_hl=max_hl,
                            ph=ph,
                            node_type_enum=eval_env.NodeType,
                            map=scene.map,
                        )
                    )

                evaluation_utils.log_batch_errors(
                    eval_batch_errors,
                    log_writer,
                    "eval",
                    epoch,
                    bar_plot=["kde"],
                    box_plot=["ade", "fde"],
                )

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_batch_errors_ml = []
                for scene in tqdm(eval_scenes, desc="MM Evaluation", ncols=80):
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions = eval_trajectron.predict(
                        scene,
                        timesteps,
                        ph,
                        num_samples=1,
                        min_future_timesteps=ph,
                        z_mode=True,
                        gmm_mode=True,
                        full_dist=False,
                    )

                    eval_batch_errors_ml.append(
                        evaluation_utils.compute_batch_statistics(
                            predictions,
                            scene.dt,
                            max_hl=max_hl,
                            ph=ph,
                            map=scene.map,
                            node_type_enum=eval_env.NodeType,
                            kde=False,
                        )
                    )

                evaluation_utils.log_batch_errors(
                    eval_batch_errors_ml, log_writer, "eval/ml", epoch
                )

        if (
            args.save_every is not None
            and args.debug is False
            and epoch % args.save_every == 0
        ):
            model_registrar.save_models(epoch)
            model_registrar.save_optimizer_and_scheduler(
                optimizer_dict=optimizer[node_type].state_dict(),
                scheduler_dict=lr_scheduler[node_type].state_dict(),
                curr_iter=epoch,
            )


if __name__ == "__main__":
    main()
