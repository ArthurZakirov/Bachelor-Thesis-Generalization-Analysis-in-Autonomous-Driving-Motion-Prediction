import sys
from tqdm import tqdm
from collections import defaultdict
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde

sys.path.append("../../Trajectron/evaluation")
sys.path.append("../../../Trajectron/trajectron")
from utils import prediction_output_to_trajectories
import visualization


def compute_tolerance_rate(predicted_trajs, gt_traj):
    num_samples, k, ph, _ = predicted_trajs.shape
    distances = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    tolerance = np.zeros((num_samples, k)).astype(float)
    for ts in range(len(gt_traj)):
        tolerance += (distances[:, :, ts] < 1.13 ** (ts + 1) - 0.9).astype(float)
    tolerance = tolerance / ph
    return tolerance.flatten()


def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.0
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(
                    kde.logpdf(gt_traj[timestep].T),
                    a_min=log_pdf_lower_bound,
                    a_max=None,
                )[0]
                kde_ll += pdf
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll / (num_timesteps * num_batches)


def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255

    interp_obs_map = RectBivariateSpline(
        range(obs_map.shape[0]), range(obs_map.shape[1]), obs_map, kx=1, ky=1
    )

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False
    )
    traj_obs_values = traj_obs_values.reshape(
        (old_shape[0], old_shape[1], old_shape[2])
    )
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)

    return num_viol_trajs


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data
    interp_obs_map = RectBivariateSpline(
        range(obs_map.shape[1]),
        range(obs_map.shape[0]),
        binary_dilation(obs_map.T, iterations=4),
        kx=1,
        ky=1,
    )

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False
    )
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_batch_statistics(
    prediction_output_dict,
    dt,
    max_hl,
    ph,
    node_type_enum,
    kde=True,
    obs=False,
    map=None,
    prune_ph_to_future=False,
    best_of=False,
    MR_radius=5,
):

    (prediction_dict, _, futures_dict) = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future
    )

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {
            "ADE": [],
            "FDE": [],
            "kde": [],
            "obs_viols": [],
            "MR": [],
            "tolerance_rate": [],
        }

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            # rediction_dict[t][node] shape -> [1, k, ph, 2]
            # utures_dict[t][node] shape ->          [ph, 2]
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            tolerance_rates = compute_tolerance_rate(
                prediction_dict[t][node], futures_dict[t][node]
            )

            batch_error_dict[node.type]["MR"].extend(
                list([1 if (fde_errors > MR_radius).all() else 0])
            )
            if kde:
                kde_ll = compute_kde_nll(
                    prediction_dict[t][node], futures_dict[t][node]
                )
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
                tolerance_rates = np.min(tolerance_rates, keepdims=True)

            batch_error_dict[node.type]["ADE"].extend(list(ade_errors))
            batch_error_dict[node.type]["FDE"].extend(list(fde_errors))
            batch_error_dict[node.type]["kde"].extend([kde_ll])
            batch_error_dict[node.type]["obs_viols"].extend([obs_viols])
            batch_error_dict[node.type]["tolerance_rate"].extend(list(tolerance_rates))
    return batch_error_dict


def evaluate_domain(eval_stg, scenes, args, env):
    eval_dict = defaultdict(dict)
    for ph in args.ph_list:
        eval_dict["ADE"][ph] = defaultdict(dict)
        eval_dict["FDE"][ph] = defaultdict(dict)
        eval_dict["MR"][ph] = defaultdict(dict)
        eval_dict["tolerance_rate"][ph] = defaultdict(dict)

    for ph_eval in tqdm(args.ph_list, desc=f"ph: {args.ph_list}", leave=False):
        with torch.no_grad():
            ############### MIN K ##############
            for k in tqdm(args.k, desc=f"k: {args.k}", leave=False):
                eval_ade_batch_errors = np.array([])
                eval_fde_batch_errors = np.array([])
                eval_mr_batch_errors = np.array([])
                eval_tr_batch_errors = np.array([])
                num_scenes = (
                    len(scenes)
                    if args.num_eval_scenes is None
                    else args.num_eval_scenes
                )
                for scene in tqdm(scenes[:num_scenes], desc="scenes", leave=False):
                    timesteps = np.arange(scene.timesteps)
                    predictions = eval_stg.predict(
                        scene,
                        timesteps,
                        ph_eval,
                        num_samples=k,
                        min_future_timesteps=ph_eval,
                        z_mode=False,
                        gmm_mode=True,
                        full_dist=False,
                        all_z_sep=False,
                        min_k=True,
                    )

                    batch_error_dict = compute_batch_statistics(
                        predictions,
                        scene.dt,
                        max_hl=args.max_hl,
                        ph=ph_eval,
                        node_type_enum=env.NodeType,
                        map=scene.map["VEHICLE"],
                        prune_ph_to_future=False,
                        kde=False,
                        best_of=True,
                        obs=False,
                    )
                    eval_ade_batch_errors = np.hstack(
                        (eval_ade_batch_errors, batch_error_dict[args.node_type]["ADE"])
                    )
                    eval_fde_batch_errors = np.hstack(
                        (eval_fde_batch_errors, batch_error_dict[args.node_type]["FDE"])
                    )
                    eval_mr_batch_errors = np.hstack(
                        (eval_mr_batch_errors, batch_error_dict[args.node_type]["MR"])
                    )
                    eval_tr_batch_errors = np.hstack(
                        (
                            eval_tr_batch_errors,
                            batch_error_dict[args.node_type]["tolerance_rate"],
                        )
                    )

                eval_dict["ADE"][ph_eval][k] = eval_ade_batch_errors.mean().item()
                eval_dict["FDE"][ph_eval][k] = eval_fde_batch_errors.mean().item()
                eval_dict["MR"][ph_eval][k] = eval_mr_batch_errors.mean().item()
                eval_dict["tolerance_rate"][ph_eval][
                    k
                ] = eval_tr_batch_errors.mean().item()
    return eval_dict


def log_batch_errors(
    batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]
):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                log_writer.add_histogram(
                    f"{node_type.name}/{namespace}/{metric}",
                    metric_batch_error,
                    curr_iter,
                )
                log_writer.add_scalar(
                    f"{node_type.name}/{namespace}/{metric}_mean",
                    np.mean(metric_batch_error),
                    curr_iter,
                )
                log_writer.add_scalar(
                    f"{node_type.name}/{namespace}/{metric}_median",
                    np.median(metric_batch_error),
                    curr_iter,
                )

                if metric in bar_plot:
                    pd = {
                        "dataset": [namespace] * len(metric_batch_error),
                        metric: metric_batch_error,
                    }
                    kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_barplots(
                        ax, pd, "dataset", metric
                    )
                    log_writer.add_figure(
                        f"{node_type.name}/{namespace}/{metric}_bar_plot",
                        kde_barplot_fig,
                        curr_iter,
                    )

                if metric in box_plot:
                    mse_fde_pd = {
                        "dataset": [namespace] * len(metric_batch_error),
                        metric: metric_batch_error,
                    }
                    fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_boxplots(
                        ax, mse_fde_pd, "dataset", metric
                    )
                    log_writer.add_figure(
                        f"{node_type.name}/{namespace}/{metric}_box_plot",
                        fig,
                        curr_iter,
                    )


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean",
                    np.mean(metric_batch_error),
                )
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median",
                    np.median(metric_batch_error),
                )


def split_by_standing_status(loc):
    return f"../split_by_characteristics/Bewegungsart/move/{loc}"


def split_by_driving_characteristics(loc):
    data_dirs = [
        f"../split_by_characteristics/Bewegungsart/stand/{loc}",
        f"../split_by_characteristics/Bewegungsart/start/{loc}",
        f"../split_by_characteristics/Bewegungsart/move/{loc}",
        f"../split_by_characteristics/Beschleunigung/const/{loc}",
        f"../split_by_characteristics/Beschleunigung/acc/{loc}",
        f"../split_by_characteristics/Beschleunigung/dec/{loc}",
        f"../split_by_characteristics/Kursaenderung/00_15/{loc}",
        f"../split_by_characteristics/Kursaenderung/15_45/{loc}",
        f"../split_by_characteristics/Kursaenderung/45_75/{loc}",
        f"../split_by_characteristics/Kursaenderung/75_360/{loc}",
        f"../split_by_characteristics/Lenkrichtung/keep/{loc}",
        f"../split_by_characteristics/Lenkrichtung/change/{loc}",
        f"../split_by_characteristics/Distanz/short/{loc}",
        f"../split_by_characteristics/Distanz/mid/{loc}",
        f"../split_by_characteristics/Distanz/long/{loc}",
    ]
    return data_dirs


def split_by_speed_zone(loc):
    data_dirs = [loc + "_slow", loc + "_middle", loc + "_fast"]
    return data_dirs


def split_by_country_or_road_class(model):
    if "Boston" in model:
        data_dirs = [
            "nuScenes_Boston_middle",
            "nuScenes_Onenorth_middle",
            "lyft_level_5_middle",
            "openDD",
            "KITTI_fast",
        ]
    elif "lyft" in model:
        data_dirs = [
            "lyft_level_5_middle",
            "nuScenes_Queenstown_middle",
            "nuScenes_Boston_middle",
            "openDD",
            "KITTI_fast",
        ]
    elif "Queenstown" in model:
        data_dirs = [
            "nuScenes_Queenstown_middle",
            "lyft_level_5_middle",
            "nuScenes_Onenorth_middle",
            "openDD",
            "KITTI_fast",
        ]
    elif "Onenorth" in model:
        data_dirs = [
            "nuScenes_Onenorth_middle",
            "nuScenes_Boston_middle",
            "nuScenes_Queenstown_middle",
            "openDD",
            "KITTI_fast",
        ]
    return data_dirs
