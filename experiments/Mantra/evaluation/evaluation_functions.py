"""
This Module has the functions for evaluate.py
"""
import sys
import torch
from torch.autograd import Variable
from tqdm import tqdm

sys.path.append("../../../Mantra/models")


def compute_batch_statistics(args, future, pred):
    """
    Arguments
    ---------
    future : torch.tensor [bs, ts, d]
    pred : torch.tensor [bs, k, ts, d]
    k : int

    Returns
    -------
    """
    future_rep = future.unsqueeze(1).repeat(1, 10, 1, 1)
    distances = torch.norm(pred - future_rep, dim=-1)  # [bs, k, ts]
    MR_boundary = 5
    batch_statistics = {
        "ADE": {
            ph: {
                k: distances[:, :k, :ph]
                .mean(dim=2)
                .min(dim=1)
                .values.mean(dim=0)
                .item()
                for k in args.k_list
            }
            for ph in args.ph_list
        },
        "FDE": {
            ph: {
                k: distances[:, :k, ph - 1].min(dim=1).values.mean(dim=0).item()
                for k in args.k_list
            }
            for ph in args.ph_list
        },
        "MR": {
            ph: {
                k: (distances[:, :k, ph - 1] > MR_boundary)
                .all(dim=-1)
                .float()
                .mean(dim=0)
                .item()
                for k in args.k_list
            }
            for ph in args.ph_list
        },
    }
    return batch_statistics


def evaluate_model(args, loader, model, device="cpu", use_map=True, write=False):
    model.model_controller.memory_past = model.memory_past
    model.model_controller.memory_fut = model.memory_fut
    print(
        f"initial memory size: {len(model.memory_past) if use_map else len(model.model_controller.memory_past)}"
    )

    model.eval()
    with torch.no_grad():
        sum_statistics = {
            "ADE": {ph: {k: 0 for k in args.k_list} for ph in args.ph_list},
            "FDE": {ph: {k: 0 for k in args.k_list} for ph in args.ph_list},
            "MR": {ph: {k: 0 for k in args.k_list} for ph in args.ph_list},
        }

        for batch in tqdm(loader, desc="batches"):
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
            ) = batch
            past = Variable(past).to(device)
            future = Variable(future).to(device)
            if use_map:
                scene_one_hot = Variable(scene_one_hot).to(device)
                pred = model(
                    past=past, future=None, scene=scene_one_hot
                )  # [bs, k, ts, d]
                if write:
                    _, _ = model(past=past, future=future, scene=scene_one_hot)

            elif not use_map:
                pred = model.model_controller(past=past)  # [bs, k, ts, d]
                if write:
                    _, _ = model.model_controller(past=past, future=future)

            batch_statistics = compute_batch_statistics(args, future, pred)
            for metric, metric_dict in batch_statistics.items():
                for ph, ph_dict in metric_dict.items():
                    for k, value in ph_dict.items():
                        sum_statistics[metric][ph][k] += value

    for metric, metric_dict in sum_statistics.items():
        for ph, ph_dict in metric_dict.items():
            for k, value in ph_dict.items():
                sum_statistics[metric][ph][k] = sum_statistics[metric][ph][k] / len(
                    loader
                )
    print(
        f"final memory size: {len(model.memory_past) if use_map else len(model.model_controller.memory_past)}"
    )
    return sum_statistics
