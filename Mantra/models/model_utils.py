import torch


def get_tolerance_rate(prediction, future, dt=0.5):
    bs, k, pred_len, _ = prediction.shape
    future_rep = future.unsqueeze(1).repeat(1, k, 1, 1) # bs, k, len, dim
    distances = torch.norm(prediction - future_rep, dim=3) # bs, k, len
    tolerance = torch.zeros((bs, k))
    tolerance = tolerance.to(prediction.device)
    for ts in range(pred_len):
        tolerance += (distances[:, :, ts] < 1.13 ** (ts + 1) - 0.9).float()
    tolerance = tolerance / pred_len
    tolerance_rate = tolerance.max(dim=1).values.type(torch.FloatTensor).unsqueeze(1)
    return tolerance_rate


def best_prediction(prediction, future):
    future_rep = future.unsqueeze(1).repeat(1, prediction.shape[1], 1, 1)
    distances = torch.norm(prediction - future_rep, dim=3)
    best_idx = distances.mean(dim=2).min(dim=1).indices
    return prediction[range(len(prediction)), best_idx]

def best_fde(prediction, future):
    future_rep = future.unsqueeze(1).repeat(1, prediction.shape[1], 1, 1)
    distances = torch.norm(prediction - future_rep, dim=3)
    best_idx = distances[:, :, -1].min(dim=1).indices
    return prediction[range(len(prediction)), best_idx]

def fde(pred, future):
    distances = torch.norm(pred - future, dim=-1)
    fdes = distances[:, -1]
    return fdes