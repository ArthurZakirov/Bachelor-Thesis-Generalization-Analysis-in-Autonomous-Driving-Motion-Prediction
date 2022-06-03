"""
Module to visualize Trajectron predictions in jupyter notebook
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.serialization import SourceChangeWarning
import warnings

warnings.filterwarnings("ignore", category=SourceChangeWarning)
sys.path.append("../../../Trajectron/trajectron")
sys.path.append("../evaluation")
sys.path.append("../experiment_utils")
sys.path.append("../train")
sys.path.append("../../../converter/converter_functions")
sys.path.append("../data_analysis")

from utils import prediction_output_to_trajectories
from experiment_utils import load_domain_env

sys.path.append("../experiment_utils")
from experiment_utils import load_model


def present_env(args, data_dir, model):
    trajectron, hyperparams = load_model(model)
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
    print("\n\nscene_id")
    print("--------")
    print(f"Wähle eine 'scene_id' aus:\n{scene_names}")
    return env, trajectron


def present_scene(env, scene_id):
    scene = get_scene_by_id(env, scene_id)
    print(f"Der Name der Szene lautet: {scene.name}")

    vehicles = [
        node
        for node in scene.nodes
        if str(node.type) == "VEHICLE" and not node.is_robot
    ]
    filled_timesteps = np.zeros(scene.timesteps)
    ph = 12
    hl = 4
    for vehicle in vehicles:
        filled_timesteps[vehicle.first_timestep + hl : vehicle._last_timestep - ph] = 1
    print(f"\ntimesteps")
    print("----------")
    print(
        f"Es gibt Fahrzeuge an allen Zeitschritten, außer {[i for i, value in enumerate(filled_timesteps) if value == 0]}."
    )
    print(f"Wähle einen Zeitschritt 0-{39} an dem es Fahrzeuge gibt.")
    return scene


def present_timestep(scene, trajectron, timesteps):
    timesteps = np.array([timesteps])
    predictions = trajectron.predict(
        scene,
        timesteps,
        ph=12,
        min_future_timesteps=12,
        min_history_timesteps=4,
        min_k=True,
        z_mode=False,
        gmm_mode=True,
        num_samples=25,
    )
    trajectories = prediction_output_to_trajectories(predictions, scene.dt, 4, 12)
    hist, fut, pred = trajectories
    vehicles = [
        node
        for node in hist[timesteps.item()].keys()
        if str(node.type) == "VEHICLE" and not node.is_robot
    ]
    print("\n\nK")
    print("-------")
    print(f"Wähle die Anzahl der visualisierten Modi 'K' aus: 0-25")

    print("\n\nnode_ids")
    print("----------")
    print(
        f"Zum gewählten Zeitschritt {timesteps.item()} gibt es {len(vehicles)} Fahrzeuge."
    )
    print(
        f"Gebe die IDs der zu visualisiererenden Fahrzeuge in die Liste 'node_ids' ein."
    )
    print(f"\n{[vehicle.id for vehicle in vehicles]}\n")
    print(f"Falls alle Fahrzeuge visualisiert werden sollen, setze: node_ids=None.")

    print("\n\nshow_robot")
    print("------------")
    print(f"Soll das Robot Fahrzeugs dargestellt werden?")

    print("\n\nshow_pedestrians")
    print("------------")
    print(f"Sollen Fußgänger dargestellt werden?")
    return scene, trajectories


def get_scene_by_id(env, scene_id):
    for scene in env.scenes:
        if scene.name == scene_id:
            return scene


def update_boundaries(boundaries, x, y):
    x_min, x_max, y_min, y_max = boundaries
    x_min = int(min(x.min().item(), x_min))
    x_max = int(max(x.max().item(), x_max))
    y_min = int(min(y.min().item(), y_min))
    y_max = int(max(y.max().item(), y_max))
    return x_min, x_max, y_min, y_max


def plot_full_scene_at_timestep(
    scene,
    trajectory_dicts,
    timestep=10,
    node_ids=None,
    k_list=range(5),
    show_robot=True,
    show_pedestrians=True,
    linewidth=2,
    scatter_size=300,
    fontsize=30,
    figsize=(30, 30),
    zoom=True,
):
    pred, hist, fut = trajectory_dicts

    # PREPARATION
    resolution = np.unique(scene.map["VEHICLE"].homography).max()
    hl = 4
    ph = 12

    vehicles = [
        node
        for node in hist[timestep].keys()
        if str(node.type) == "VEHICLE" and not node.is_robot
    ]
    if not node_ids is None:
        vehicles = [vehicle for vehicle in vehicles if vehicle.id in node_ids]
    pedestrians = [node for node in scene.nodes if str(node.type) == "PEDESTRIAN"]

    # DETERMINE BOUNDARIES FOR ZOOMING
    boundaries = 100000, 0, 100000, 0
    if show_robot:
        robot_node_data = scene.robot.get(
            np.array([timestep - hl, timestep + ph]), {"position": {"x", "y"}}
        )
        x_robot, y_robot = resolution * robot_node_data.T[::-1]
        boundaries = update_boundaries(boundaries, x=x_robot, y=y_robot)

    if show_pedestrians:
        for pedestrian in pedestrians:
            x_ped, y_ped = pedestrian.data.data[:, :2].T
            boundaries = update_boundaries(boundaries, x=x_ped, y=y_ped)

    for node in vehicles:
        x_hist, y_hist = resolution * hist[timestep][node].T
        boundaries = update_boundaries(boundaries, x=x_hist, y=y_hist)

        x_fut, y_fut = resolution * fut[timestep][node].T
        x_fut = np.insert(x_fut, 0, x_hist[-1])
        y_fut = np.insert(y_fut, 0, y_hist[-1])
        boundaries = update_boundaries(boundaries, x=x_fut, y=y_fut)

    if zoom:
        buffer = 50
        x_min, x_max, y_min, y_max = boundaries

        x_min = x_min - buffer
        x_max = x_max + buffer
        y_min = y_min - buffer
        y_max = y_max + buffer

        shift = np.array([[x_min], [y_min]])
    else:
        shift = np.zeros((2, 1))
    # PLOT
    fig, ax = plt.subplots(figsize=figsize)

    for node in vehicles:
        x_hist, y_hist = resolution * hist[timestep][node].T - shift
        ax.scatter(x_hist[-1], y_hist[-1], color="b", s=scatter_size)
        ax.plot(x_hist, y_hist, "b", linewidth=linewidth)

        x_fut, y_fut = resolution * fut[timestep][node].T - shift
        x_fut = np.insert(x_fut, 0, x_hist[-1])
        y_fut = np.insert(y_fut, 0, y_hist[-1])
        ax.plot(x_fut, y_fut, "r", linewidth=linewidth)

        for k in k_list:
            x_pred, y_pred = resolution * pred[timestep][node].squeeze(0)[k].T - shift
            x_pred = np.insert(x_pred, 0, x_hist[-1])
            y_pred = np.insert(y_pred, 0, y_hist[-1])
            ax.plot(x_pred[:ph], y_pred[:ph], "c--", linewidth=linewidth)

    if show_robot:
        robot_node_data = scene.robot.get(
            np.array([timestep - hl, timestep + ph]), {"position": {"x", "y"}}
        )
        x_robot, y_robot = resolution * robot_node_data.T[::-1] - shift
        ax.scatter(x_robot[hl + 1], y_robot[hl + 1], color="y", s=scatter_size)
        ax.plot(x_robot, y_robot, "y", linewidth=linewidth, label="robot")

    if show_pedestrians:
        for pedestrian in pedestrians:
            x_ped, y_ped = pedestrian.data.data[:, :2].T - shift
            plt.scatter(x_ped, y_ped, color="g")

    scene_mask = scene.map["VEHICLE"].torch_map("cpu")
    ax.imshow(
        ~scene_mask[0, x_min:x_max, y_min:y_max].T,
        cmap="Greys",
        interpolation="nearest",
    )
    ax.plot(0, 0, "c--", label="Prädiktion")
    ax.plot(0, 0, "b", label="Vergangenheit")
    ax.plot(0, 0, "r", label="Zukunft")
    ax.plot(0, 0, "g", label="Fußgänger")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    return fig, ax
