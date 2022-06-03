"""
Module that contains functions for MANTRA IRM Training
"""
import os
import sys
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from models.model_memory_IRM import model_memory_IRM, model_controllerMem
import index_qualitative

sys.path.append("../../../Mantra/models")
sys.path.append("../experiment_utils")
from model_utils import best_prediction
from experiment_utils import load_model
from .trainer_controllerMem import TrainerController


class TrainerIRM(TrainerController):
    def __init__(self, config, dataset_and_loader=None):
        """
        Trainer class for training the Iterative Refinement Module (IRM)
        :param config: configuration parameters (see train_IRM.py)
        """
        self.index_qualitative = index_qualitative.dict_test
        self.name_test = config.time
        model_dir = os.path.join(
            config.experiment_tag,
            config.data_dir.split("_middle")[0]
            + ("" if config.time is None else ("_" + config.time)),
        )
        self.folder_test = os.path.join("../models/training_IRM/", model_dir)
        self.name_run = self.folder_test
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.file = open(os.path.join(self.folder_test, "details.txt"), "w")

        (
            self.data_train,
            self.train_loader,
            self.data_test,
            self.test_loader,
        ) = dataset_and_loader

        self.dim_clip = config.dim_clip
        self.num_prediction = config.preds
        self.device = config.device
        self.settings = {
            "batch_size": config.batch_size,
            "device": config.device,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": self.num_prediction,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "dim_clip": config.dim_clip,
        }
        self.max_epochs = config.max_epochs
        # load pretrained model and create memory_model

        model_ae = load_model(part="training_ae", model_dir=model_dir)
        model_controller = model_controllerMem(self.settings, model_ae)

        self.mem_n2n, self.last_epoch = (
            load_model(part="training_IRM", model_dir=model_dir, return_checkpoint=True)
            if config.resume
            else (model_memory_IRM(self.settings, model_controller), 0)
        )
        self.mem_n2n.past_len = config.past_len
        self.mem_n2n.future_len = config.future_len

        self.criterionLoss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        self.criterionLoss = self.criterionLoss.to(self.device)
        self.mem_n2n = self.mem_n2n.to(self.device)
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.name_run)
        self.writer.add_text(
            "Training Configuration", "model name: " + self.mem_n2n.name_model, 0
        )
        self.writer.add_text(
            "Training Configuration", "dataset train: " + str(len(self.data_train)), 0
        )
        self.writer.add_text(
            "Training Configuration", "dataset test: " + str(len(self.data_test)), 0
        )
        self.writer.add_text(
            "Training Configuration",
            "number of prediction: " + str(self.num_prediction),
            0,
        )
        self.writer.add_text(
            "Training Configuration", "batch_size: " + str(self.config.batch_size), 0
        )
        self.writer.add_text(
            "Training Configuration",
            "learning rate init: " + str(self.config.learning_rate),
            0,
        )
        self.writer.add_text(
            "Training Configuration",
            "dim_embedding_key: " + str(self.settings["dim_embedding_key"]),
            0,
        )

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """
        self.file.write("points of past track: " + str(self.config.past_len) + "\n")
        self.file.write("points of future track: " + str(self.config.future_len) + "\n")
        self.file.write("train size: " + str(len(self.data_train)) + "\n")
        self.file.write("test size: " + str(len(self.data_test)) + "\n")
        self.file.write("batch size: " + str(self.config.batch_size) + "\n")

    def fit(self):
        """
        Iterative refinement model training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.model_ae.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.model_ae.conv_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.model_ae.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.model_ae.encoder_fut.parameters():
            param.requires_grad = False

        self.mem_n2n.model_ae_pre_IRM_training = deepcopy(self.mem_n2n.model_ae)

        for param in self.mem_n2n.model_ae.decoder.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.model_ae.FC_output.parameters():
            param.requires_grad = True

        for param in self.mem_n2n.model_controller.linear_controller.parameters():
            param.requires_grad = False

        for param in self.mem_n2n.convScene_1.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.convScene_2.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.RNN_scene.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.fc_refine.parameters():
            param.requires_grad = True

        # Load memory
        # populate the memory
        if not config.resume:
            self._memory_writing(self.config.saved_memory)
            self._save_memory()
        self.writer.add_text(
            "Training Configuration",
            "memory size: " + str(len(self.mem_n2n.memory_past)) + "\n",
        )
        print(f"MEMORY SIZE: {len(self.mem_n2n.memory_past)}")
        # Main training loop
        for epoch in tqdm(range(self.last_epoch + 1, config.max_epochs)):
            self.mem_n2n.train()
            self._train_single_epoch(epoch)

            if epoch % self.config.eval_every == 0:
                self.evaluate(self.test_loader, epoch)
                torch.save(
                    self.mem_n2n, os.path.join(self.folder_test, f"model-{epoch}")
                )

    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """
        sum_loss = 0
        self.mem_n2n.eval()
        with torch.no_grad():
            for (
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
            ) in tqdm(loader, leave=False, desc="eval"):
                past = Variable(past)
                future = Variable(future)
                scene_one_hot = Variable(scene_one_hot)
                past = past.to(self.device)
                future = future.to(self.device)
                scene_one_hot = scene_one_hot.to(self.device)
                output = self.mem_n2n(past, scene=scene_one_hot)
                best_pred = best_prediction(output, future)
                sum_loss += self.criterionLoss(best_pred, future).detach()
        mean_loss = sum_loss / len(loader)
        self.writer.add_scalar("eval/loss", mean_loss, epoch)

    def _train_single_epoch(self, epoch):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        sum_loss = 0
        sum_min_ADE_k = 0
        sum_min_ADE_k_no_map = 0
        sum_loss_no_map = 0
        self.mem_n2n.train()

        for (index, past, future, _, _, _, _, _, _, scene_one_hot) in tqdm(
            self.train_loader, desc=f"batches", leave=False
        ):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            scene_one_hot = Variable(scene_one_hot)
            past = past.to(self.device)
            future = future.to(self.device)
            scene_one_hot = scene_one_hot.to(self.device)
            self.opt.zero_grad()
            prediction = self.mem_n2n(past=past, scene=scene_one_hot)
            prediction_no_map = self.mem_n2n(past=past)

            best_pred = best_prediction(prediction, future)
            best_pred_no_map = best_prediction(prediction_no_map, future)
            loss = self.criterionLoss(best_pred, future)
            loss_no_map = self.criterionLoss(best_pred_no_map, future)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            sum_loss += loss.detach()
            sum_loss_no_map += loss_no_map.detach()
            sum_min_ADE_k += (
                torch.norm(best_pred - future, dim=2).mean(dim=1).mean(dim=0).detach()
            )
            sum_min_ADE_k_no_map += (
                torch.norm(best_pred_no_map - future, dim=2)
                .mean(dim=1)
                .mean(dim=0)
                .detach()
            )

        mean_loss = sum_loss / len(self.train_loader)
        mean_loss_no_map = sum_loss_no_map / len(self.train_loader)
        mean_min_ADE_k = sum_min_ADE_k / len(self.train_loader)
        mean_min_ADE_k_no_map = sum_min_ADE_k_no_map / len(self.train_loader)
        self.writer.add_scalars(
            "train/train_loss",
            {"with IRM": mean_loss, "without IRM": mean_loss_no_map},
            epoch,
        )
        self.writer.add_scalars(
            "train/min_ADE_k",
            {"with IRM": mean_min_ADE_k, "without IRM": mean_min_ADE_k_no_map},
            epoch,
        )
        self.writer.add_scalar(
            "Training Configuration/memory size ", len(self.mem_n2n.memory_past), 0
        )

    def _save_memory(self):
        torch.save(
            self.mem_n2n.memory_past, os.path.join(self.folder_test, "memory_past.pt")
        )
        torch.save(
            self.mem_n2n.memory_fut, os.path.join(self.folder_test, "memory_fut.pt")
        )
