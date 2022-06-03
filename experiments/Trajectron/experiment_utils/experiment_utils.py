"""
In this module are utils used in Training / Evaluation / Visualization of Trajectron++
Loading model, Loading data etc.
"""
import sys
import os
import json
import warnings
from tqdm import tqdm
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)
sys.path.append("../../../Trajectron/trajectron")
sys.path.append("../../Trajectron/trajectron")
sys.path.append("../../../converter/converter_functions")
sys.path.append("../converter_functions")
sys.path.append("../../Trajectron/train")
sys.path.append("../train")
sys.path.append("../../Trajectron/data_analysis")
sys.path.append("../data_analysis")
sys.path.append("../../experiments/Trajectron/train")
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from train_utils import prepare_dataset
from data2env_functions import init_env


def load_model(model_dir, epoch=None, return_last_checkpoint=False):
    """
    Load Trajectron++

    :param model_dir: directory of model
    :param ts: training step of the model (epoch)
    :param return_last_checkpoint: if True, function returns int of last epoch
    :return: trajectron, hyperparams, max_checkpoint
    """
    if not ("experiments" in model_dir):
        model_dir = os.path.join("../models", model_dir)
    if epoch is None:
        max_checkpoint = 0
        for file in os.listdir(model_dir):
            if "model_registrar" in file:
                curr_checkpoint = int(file.split(".")[0].split("-")[1])
                if curr_checkpoint > max_checkpoint:
                    max_checkpoint = curr_checkpoint
        checkpoint = max_checkpoint
    else:
        checkpoint = epoch
    model_registrar = ModelRegistrar(model_dir, "cpu")
    model_registrar.load_models(checkpoint)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, "cpu")
    if not return_last_checkpoint:
        output = trajectron, hyperparams
    else:
        output = trajectron, hyperparams, max_checkpoint
    return output


def load_domain_env(args, hyperparams, data_dir, train=False, speed=""):
    """
    Loads all environments from a given data_dir and combines them into a single environment.

    :param args:
    :param hyperparams:
    :param data_dir: directory with data
    :param train: if True, training data is loaded, else eval data
    :param speed: exclude domains that don't fall in to the given speed
    :return: domain_env
    """
    if not "processed" in data_dir:
        data_dir = os.path.join(
            "../../../datasets/processed/Trajectron_format", data_dir
        )
    domain_env = init_env()
    domain_env.scenes = []
    domain_scenes = []
    for data_file in os.listdir(data_dir):
        if not train and not (
            "val" in data_file or "eval" in data_file or "test" in data_file
        ):
            continue
        if train and not "train" in data_file:
            continue
        if not speed in data_file:
            continue
        data_path = os.path.join(data_dir, data_file)
        (_, _, _, scenes) = prepare_dataset(data_path, hyperparams, args)
        domain_scenes = domain_scenes + scenes
    for scene in tqdm(domain_scenes):
        scene.calculate_scene_graph(
            domain_env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )
        domain_env.scenes.append(scene)
    return domain_env
