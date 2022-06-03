"""
Module that contains functions for MANTRA Visualization in Jupyter Notebook
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import cv2

sys.path.append("../evaluation")
sys.path.append("../..")

sys.path.append("../../Trajectron/experiment_utils")
from Trajectron.experiment_utils import load_model as load_Trajectron
from Trajectron.experiment_utils import load_domain_env


def get_scene_by_id(env, scene_id):
    for scene in env.scenes:
        if scene.name == scene_id:
            return scene


def present_env(args, data_dir, model):
    trajectron, hyperparams = load_Trajectron(model)
    env = load_domain_env(args, hyperparams, data_dir, train=False)
    trajectron.set_environment(env)
    scenes = [
        scene
        for scene in env.scenes
        if scene.name
        in [
            scene.name
            for scene in env.scenes
            for node in scene.nodes
            if not node.is_robot and str(node.type) == "VEHICLE"
        ]
    ]
    scene_names = [scene.name for scene in scenes]
    print("\n\n\n\nscene_id")
    print("--------")
    print(f"Wähle eine 'scene_id' aus:\n{scene_names}")
    return env, trajectron


def present_scene(env, scene_id):
    scene = get_scene_by_id(env, scene_id)
    print(f"\n\n\n\nDer Name der Szene lautet: {scene.name}")

    vehicles = [
        node
        for node in scene.nodes
        if str(node.type) == "VEHICLE"
        and not node.is_robot
        and node._last_timestep - node.first_timestep > 12 + 4
    ]
    print(f"\n\n\n\nnode_id")
    print("----------")
    print(f"In der Szene gibt es {len(vehicles)} Fahrzeuge.")
    print(f"\n{[vehicle.id for vehicle in vehicles]}\n")
    print(f"Wähle die node_id des zu visualisierenden Fahrzeugs")
    return scene


def present_node(scene, node_id):
    ph = 12
    hl = 4
    for node in scene.nodes:
        if node.id == node_id:
            break
    print("\n\n\n\ntimestep")
    print("--------------")
    print(
        f"node {node} erscheint zwischen den Zeitschritten {node.first_timestep+hl}-{node._last_timestep-ph}."
    )
    print(f"Gebe den zu visualisierenden Zeitschritt ein.")
    return node


def sample_np2torch(sample):
    return [
        torch.from_numpy(element) if (type(element) == np.ndarray) else element
        for element in sample
    ]


def plot_sample(
    mem_n2n,
    args,
    sample,
    mode="IRM",
    write=False,
    k_max=5,
    figsize=(20, 20),
    linewidth=2,
    scatter_size=300,
    fontsize=25,
):

    device = args.device
    (
        index,
        past,
        future,
        presents,
        angle_presents,
        videos,
        vehicles,
        number_vec,
        scene,
        scene_one_hot,
    ) = sample

    if mode == "IRM":
        scene_one_hot = Variable(scene_one_hot).to(device)
        pred = mem_n2n(
            past=past.unsqueeze(0), scene=scene_one_hot.unsqueeze(0)
        )  # [bs, k, ts, d]
        if write:
            _, _ = mem_n2n(
                past=past.unsqueeze(0),
                future=future.unsqueeze(0),
                scene=scene_one_hot.unsqueeze(0),
            )
    else:
        pred = mem_n2n(past=past.unsqueeze(0))

    pred = pred[0].detach()

    homography = 2
    matRot_track = cv2.getRotationMatrix2D((0, 0), -angle_presents, 1)
    past_sample = cv2.transform(past.numpy().reshape(-1, 1, 2), matRot_track).squeeze()
    future_sample = cv2.transform(
        future.numpy().reshape(-1, 1, 2), matRot_track
    ).squeeze()
    pred_sample = np.stack(
        [
            cv2.transform(pred_k.numpy().reshape(-1, 1, 2), matRot_track).squeeze()
            for pred_k in pred
        ]
    )
    past_scene = past_sample * homography + args.dim_clip
    fut_scene = future_sample * homography + args.dim_clip
    pred_scene = pred_sample * homography + args.dim_clip
    fut_scene = np.concatenate([past_scene[-1][np.newaxis], fut_scene], axis=0)
    pred_scene = np.concatenate(
        [np.tile(past_scene[-1], reps=(len(pred_scene), 1, 1)), pred_scene], axis=1
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    ax.plot(past_scene[:, 0], past_scene[:, 1], "b", linewidth=linewidth)
    ax.scatter(past_scene[-1, 0], past_scene[-1, 1], color="b", s=scatter_size)
    ax.plot(fut_scene[:, 0], fut_scene[:, 1], "r", linewidth=linewidth)

    for k, pred_scene_k in enumerate(pred_scene):
        if k > k_max:
            continue
        ax.plot(
            pred_scene_k[:, 0], pred_scene_k[:, 1], "--", color="c", linewidth=linewidth
        )

    ax.plot(fut_scene[:, 0], fut_scene[:, 1], "r", linewidth=linewidth)
    ax.imshow(~scene.bool(), cmap="Greys", interpolation="nearest")

    ax.plot(0, 0, "c", label="prediction")
    ax.plot(0, 0, "b", label="past")
    ax.plot(0, 0, "r", label="future")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    return mem_n2n
