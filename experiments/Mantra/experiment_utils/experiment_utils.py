"""
Module that contains util functions for MANTRA experiments
"""
import os
import sys
import dill
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader

sys.path.append("../../../Mantra")
sys.path.append("../../../Trajectron/trajectron")
sys.path.append("../../../converter/converter_main")
sys.path.append("../../../converter/converter_functions")
sys.path.append("../../../converter/converter_maps")
sys.path.append("../../../Trajectron/trajectron")
sys.path.append("../../Trajectron/train")
from dataset_invariance import TrackDataset
from preprocessing_utils import resize_map_mask, normalize_trajectory
from train_utils import shorten_dataset, shuffle_dataset, concat_datasets

data_columns_vehicle = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration", "heading"], ["x", "y"]]
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")])
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]])
)


def env_scene2df_samples(args, scene, scene_id):
    """
    Arguments
    ---------
    hl : float
        history length, measured in timesteps (not seconds)
    fl : float
        future horizon, measured in timesteps (not seconds)
    dt : flat
        sampling time, measured in seconds

    Returns
    -------
    df_samples: pd.DataFrame
        timestepwise samples of (hist, fut, relative position in map, relative angle in map, categorical_features)
    """
    df_samples = pd.DataFrame()
    sample_id = 0
    num_vec = 0
    hl = args.past_len
    fl = args.future_len
    for node in tqdm(scene.nodes, desc="process node", leave=False):
        if args.node_id is not None and node.id != args.node_id:
            continue
        if not str(node.type) == "VEHICLE":
            continue
        if node.is_robot and not args.use_robot_trajectories:
            continue
        node_df = pd.DataFrame(node.data.data)
        node_df.columns = data_columns_vehicle
        node_df["node_id"] = node.id
        node_df["type"] = "Car"

        xy = node_df["position"].values

        # remove nodes with less than 8 sec observation
        if not (len(node_df) >= hl + fl + 1):
            continue

        for ts in range(hl, len(node_df) - fl):
            xy_norm, course, pos = normalize_trajectory(xy, ts)
            hist = xy_norm[ts - hl : (ts + 1)]
            fut = xy_norm[(ts + 1) : (ts + 1) + fl]
            sample = {
                "index": ts,
                "pasts": [hist.tolist()],
                "futures": [fut.tolist()],
                "presents": [pos.tolist()],
                "angle_presents": -course.item(),
                "video_track": scene.name,
                "vehicles": "Car",
                "number_vec": num_vec,
                "scene_id": scene_id,
            }
            df_sample = pd.DataFrame(index=[sample_id], data=sample)
            df_samples = df_samples.append(df_sample)

            sample_id += 1
        num_vec += 1
    return df_samples


def env2Kitti(args, env, env_id, train, data_file):
    df_samples_all_scenes = []
    scene_mask_dict = dict()
    scene_idx = 0
    num_scenes = len(
        env.scenes
    )  #  120 if (not train and "lyft_level_5" in data_file) else l
    for scene in tqdm(env.scenes[:num_scenes], desc="scenes", leave=False):
        if args.scene_name is not None and scene.name != args.scene_name:
            continue

        scene_idx += 1
        scene_id = f"{env_id}-{scene_idx}"
        df_samples = env_scene2df_samples(args, scene, scene_id)
        scene_mask = scene.map["VEHICLE"].torch_map("cpu").numpy().squeeze().T
        scene_mask = resize_map_mask(scene_mask, 2 / 3) / 255
        scene_mask_dict[scene_id] = scene_mask
        df_samples_all_scenes.append(df_samples)
    if len(df_samples_all_scenes) > 0:
        df_samples_all_scenes = pd.concat(df_samples_all_scenes, axis=0)
    else:
        df_samples_all_scenes = pd.DataFrame()
    return df_samples_all_scenes, scene_mask_dict


def load_data_from_env(config, data_dir, train):
    print(f'\n\npreprocess {"train" if train else "eval"} data: {data_dir}')
    train_data_dir = os.path.join(
        "../../../datasets/processed/Trajectron_format", data_dir
    )
    train_datasets = []
    env_id = 0
    for data_file in tqdm(os.listdir(train_data_dir), desc="environments"):
        if train and not ("train" in data_file):
            continue
        elif not train and not (
            "test" in data_file or "eval" in data_file or "val" in data_file
        ):
            continue
        if not (config.speed in data_file):
            continue
        with open(os.path.join(train_data_dir, data_file), "rb") as f:
            env = dill.load(f, encoding="latin1")
        env_id += 1
        df_samples, scene_mask_dict = env2Kitti(
            args=config, env=env, env_id=env_id, train=train, data_file=data_file
        )

        df_samples.columns = pd.MultiIndex.from_product(
            [[data_file], df_samples.columns]
        )
        if df_samples.empty:
            continue
        train_dataset = TrackDataset.from_df(
            df_samples,
            scene_mask_dict,
            len_past=config.past_len,
            len_future=config.future_len,
            dt=config.dt,
            dim_clip=config.dim_clip,
        )
        train_datasets.append(train_dataset)

    if len(train_datasets) > 1:
        train_dataset = concat_datasets(
            dataset_list=train_datasets,
            num_samples_total=config.num_train_samples
            if train
            else config.num_eval_samples,
            percentages=config.percentages,
        )
        train_dataset = shuffle_dataset(train_dataset)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        return None, None
    print(
        f"{'train' if train else 'eval'} dataset size, before shorten: {len(train_dataset)}"
    )
    train_dataset = shorten_dataset(
        dataset=train_dataset,
        num_samples=config.num_train_samples if train else config.num_eval_samples,
    )
    print(
        f"{'train' if train else 'eval'} dataset size, after shorten: {len(train_dataset)}\n\n\n\n"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.preprocess_workers,
        shuffle=config.shuffle_loader,
    )
    return train_dataset, train_dataloader


def dataset_sample_plots(dataset, dim_clip=180):
    i = 0
    num_plots = 4
    fig, ax = plt.subplots(1, num_plots, figsize=(25, 25))
    for _ in range(num_plots):
        sample = dataset[i * 10]
        (
            index,
            past,
            future,
            presents,
            angle,
            video_track,
            vehicles,
            number_vec,
            scene,
            scene_crop,
        ) = sample
        matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        past = cv2.transform(
            past.cpu().numpy().reshape(-1, 1, 2), matRot_track
        ).squeeze()
        future = cv2.transform(
            future.cpu().numpy().reshape(-1, 1, 2), matRot_track
        ).squeeze()
        past_scene = past * 2 + dim_clip
        fut_scene = future * 2 + dim_clip
        if vehicles == "Car":
            ax[i].imshow(scene)
            ax[i].plot(past_scene[:, 0], past_scene[:, 1], label="past", linewidth=4)
            ax[i].plot(fut_scene[:, 0], fut_scene[:, 1], label="future", linewidth=4)
            ax[i].legend(loc="best")
            ax[i].set_title(vehicles)
            i += 1
    plt.show()
    return fig


def load_model(part, model_dir, checkpoint=None, return_checkpoint=False):
    model_dir = os.path.join(f"../models/{part}", model_dir)
    if checkpoint is None:
        checkpoint = 0
        for file in os.listdir(model_dir):
            if "model" in file:
                curr_checkpoint = int(file.split("-")[1])
                if curr_checkpoint > checkpoint:
                    checkpoint = curr_checkpoint
    model_path = os.path.join(model_dir, f"model-{checkpoint}")
    model = torch.load(model_path, map_location=torch.device("cpu"))
    load_memory_info = ""
    if part == "training_IRM":
        use_updated_memory = "memory_past_updated.pt" in os.listdir(model_dir)
        if use_updated_memory:
            memory_past_file = "memory_past_updated.pt"
            memory_fut_file = "memory_fut_updated.pt"
        else:
            memory_past_file = "memory_past.pt"
            memory_fut_file = "memory_fut.pt"
        model.memory_past = torch.load(os.path.join(model_dir, memory_past_file))
        model.memory_fut = torch.load(os.path.join(model_dir, memory_fut_file))
        load_memory_info = (
            f" with {'not' if not use_updated_memory else ''} updated memory."
        )
    print(f"Loaded {part} from {model_dir}" + load_memory_info)
    if return_checkpoint:
        output = model, checkpoint
    else:
        output = model
    return output
