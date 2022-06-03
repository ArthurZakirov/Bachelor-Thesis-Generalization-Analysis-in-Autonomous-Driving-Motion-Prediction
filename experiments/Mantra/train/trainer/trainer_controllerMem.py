"""
Module that contains functions for MANTRA Controller Training
"""
import os
import sys
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from models.model_controllerMem import model_controllerMem
from torch.autograd import Variable
import io
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

sys.path.append("../../../Mantra/models")
sys.path.append("../experiment_utils")
from experiment_utils import load_model

from model_utils import best_prediction


class TrainerController:
    def __init__(self, config, dataset, time, dataset_and_loader=None):
        """
        The Trainer class handles the training procedure for training the memory writing controller.
        :param config: configuration parameters (see train_controllerMem.py)
        """

        self.name_test = time
        self.folder_test = os.path.join(
            "../models/training_controller", dataset + "_" + time
        )
        self.folder_tensorboard = self.folder_test
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.file = open(os.path.join(self.folder_test, "details.txt"), "w")

        (
            self.data_train,
            self.train_loader,
            self.data_test,
            self.test_loader,
        ) = dataset_and_loader

        self.device = config.device
        self.dim_clip = config.dim_clip
        self.settings = {
            "batch_size": config.batch_size,
            "device": config.device,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.preds,
            "past_len": config.past_len,
            "future_len": config.future_len,
        }
        self.max_epochs = config.max_epochs
        # load pretrained model and create memory model
        self.model_ae, _ = load_model(
            part="training_ae", model_dir=dataset + "_" + time
        )
        controller_dir = None
        self.mem_n2n, _ = (
            load_model(part="training_controller", model_dir=controller_dir)
            if controller_dir is not None
            else (model_controllerMem(self.settings, self.model_ae), 0)
        )
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.5)
        self.iterations = 0
        self.mem_n2n = self.mem_n2n.to(self.device)
        self.start_epoch = 0
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_tensorboard)
        self.writer.add_text(
            "Training Configuration",
            "model name: {}".format(self.mem_n2n.name_model),
            0,
        )
        self.writer.add_text(
            "Training Configuration",
            "dataset train: {}".format(len(self.data_train)),
            0,
        )
        self.writer.add_text(
            "Training Configuration", "dataset test: {}".format(len(self.data_test)), 0
        )
        self.writer.add_text(
            "Training Configuration", "batch_size: {}".format(self.config.batch_size), 0
        )
        self.writer.add_text(
            "Training Configuration",
            "learning rate init: {}".format(self.config.learning_rate),
            0,
        )
        self.writer.add_text(
            "Training Configuration",
            "dim_embedding_key: {}".format(self.config.dim_embedding_key),
            0,
        )

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """

        self.file.write("points of past track: {}".format(self.config.past_len) + "\n")
        self.file.write(
            "points of future track: {}".format(self.config.future_len) + "\n"
        )
        self.file.write("train size: {}".format(len(self.data_train)) + "\n")
        self.file.write("test size: {}".format(len(self.data_test)) + "\n")
        self.file.write("batch size: {}".format(self.config.batch_size) + "\n")
        self.file.write("learning rate: {}".format(self.config.learning_rate) + "\n")
        self.file.write(
            "embedding dim: {}".format(self.config.dim_embedding_key) + "\n"
        )

    def fit(self):
        """
        Writing controller training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.conv_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        # Memory Initialization
        self.mem_n2n.init_memory(self.data_train)
        self.save_plot_controller(0)

        # Main training loop
        for epoch in tqdm(
            range(self.start_epoch, config.max_epochs), desc="epochs", leave=True
        ):
            self.mem_n2n.init_memory(self.data_train)
            self._train_single_epoch(epoch)
            self.save_plot_controller(epoch)

            if epoch % self.config.eval_every == 0:
                self.evaluate(self.test_loader, epoch)
                torch.save(
                    self.mem_n2n, os.path.join(self.folder_test, f"model-{epoch}")
                )

        # Save final trained model
        # torch.save(self.mem_n2n, os.path.join(self.folder_test, 'model'))

    def save_plot_controller(self, epoch):
        """
        plot the learned threshold bt writing controller
        :param epoch: epoch index (default: 0)
        :return: None
        """

        fig = plt.figure()
        x = torch.Tensor(np.linspace(0, 1, 100))
        weight = self.mem_n2n.linear_controller.weight.cpu()
        bias = self.mem_n2n.linear_controller.bias.cpu()
        y = torch.sigmoid(weight * x + bias).squeeze()
        plt.plot(
            x.data.numpy(),
            y.data.numpy(),
            "-r",
            label="y=" + str(weight.item()) + "x + " + str(bias.item()),
        )
        plt.plot(x.data.numpy(), [0.5] * 100, "-b")
        plt.title("controller")
        plt.axis([0, 1, 0, 1])
        plt.xlabel("x", color="#1C2833")
        plt.ylabel("y", color="#1C2833")
        plt.legend(loc="upper left")
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)

        self.writer.add_image("controller_plot/function", image.squeeze(0), epoch)
        plt.close(fig)

    def plot_track(
        self,
        past,
        future,
        scene,
        pred=None,
        angle=0,
        video_id="",
        vec_id="",
        index_tracklet=0,
        num_epoch=0,
    ):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param scene: the observed scene where is the trajectory
        :param pred: predicted future trajectory
        :param angle: rotation angle to plot the trajectory in the original direction
        :param video_id: video index of the trajectory
        :param vec_id: vehicle type of the trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :return: None
        """

        colors = [
            (0, 0, 0),
            (0.87, 0.87, 0.87),
            (0.54, 0.54, 0.54),
            (0.49, 0.33, 0.16),
            (0.29, 0.57, 0.25),
        ]
        cmap_name = "scene_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)

        fig = plt.figure()
        plt.imshow(scene)
        colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))

        matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        past = cv2.transform(
            past.cpu().numpy().reshape(-1, 1, 2), matRot_track
        ).squeeze()
        future = cv2.transform(
            future.cpu().numpy().reshape(-1, 1, 2), matRot_track
        ).squeeze()
        past_scene = past * 2 + self.dim_clip
        future_scene = future * 2 + self.dim_clip
        plt.plot(
            past_scene[:, 0],
            past_scene[:, 1],
            c="blue",
            linewidth=1,
            marker="o",
            markersize=1,
        )
        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                pred_i = cv2.transform(
                    pred[i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track
                ).squeeze()
                pred_scene = pred_i * 2 + self.dim_clip
                plt.plot(
                    pred_scene[:, 0],
                    pred_scene[:, 1],
                    color=colors[i_p],
                    linewidth=0.5,
                    marker="o",
                    markersize=0.5,
                )
        plt.plot(
            future_scene[:, 0],
            future_scene[:, 1],
            c="green",
            linewidth=1,
            marker="o",
            markersize=1,
        )
        plt.title(
            "video: "
            + video_id
            + ", vehicle: "
            + vec_id
            + ", index: "
            + str(index_tracklet)
        )
        plt.axis("equal")

        # Save figure in Tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_image(
            "Image_test/track_"
            + video_id
            + "_"
            + vec_id
            + "_"
            + str(index_tracklet).zfill(3),
            image.squeeze(0),
            num_epoch,
        )
        plt.close(fig)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """
        self._memory_writing()
        sum_controller_loss = 0
        sum_pred_loss = 0
        with torch.no_grad():
            for step, (
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
            ) in enumerate(loader):
                past = Variable(past)
                future = Variable(future)
                past = past.to(self.device)
                future = future.to(self.device)
                pred = self.mem_n2n(past)
                prob, sim = self.mem_n2n(past, future)
                sum_controller_loss += self.ControllerLoss(prob, sim).detach()
                best_pred = best_prediction(pred, future)
                sum_pred_loss += self.criterionLoss(best_pred, future)

        mean_controller_loss = sum_controller_loss / len(loader)
        mean_pred_loss = sum_pred_loss / len(loader)
        self.writer.add_scalar("eval/controller_loss", mean_controller_loss, epoch)
        self.writer.add_scalar("eval/pred_loss", mean_pred_loss, epoch)

        idx = 7
        (
            index,
            past,
            future,
            present,
            angle_present,
            video_track,
            vehicles,
            number_vec,
            scene,
            _,
        ) = self.data_train[idx]
        self.plot_track(
            past,
            future,
            scene,
            pred,
            angle_present,
            video_track,
            vehicles,
            number_vec,
            epoch,
        )

    def _train_single_epoch(self, epoch):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        sum_controller_loss = 0
        sum_pred_loss = 0
        config = self.config
        for step, (index, past, future, _, _, _, _, _, _, _) in enumerate(
            tqdm(self.train_loader, desc="batches", leave=False)
        ):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            past = past.to(self.device)
            future = future.to(self.device)
            self.opt.zero_grad()
            prob, sim = self.mem_n2n(past, future)

            pred = self.mem_n2n(past)
            controller_loss = self.ControllerLoss(prob, sim)
            best_pred = best_prediction(pred, future)
            pred_loss = self.criterionLoss(best_pred, future)
            min_ADE_k = torch.norm(best_pred - future, dim=2).mean(dim=1).mean(dim=0)
            controller_loss.backward()
            self.opt.step()
            batch_id = epoch * len(self.train_loader) + step
            self.writer.add_scalar(
                "train/controller_loss", controller_loss.detach(), batch_id
            )
            self.writer.add_scalar("train/pred_loss", pred_loss.detach(), batch_id)
            self.writer.add_scalar(
                "train/tolerance_rate", sim.mean().detach(), batch_id
            )
            self.writer.add_scalar("train/prob", prob.mean().detach(), batch_id)
            self.writer.add_scalar(
                "train/memory_size", len(self.mem_n2n.memory_past), batch_id
            )
            self.writer.add_scalar("train/min_ADE_k", min_ADE_k.detach(), batch_id)

            # sum_controller_loss += controller_loss.detach()
            # sum_pred_loss += pred_loss.detach()

        # mean_controller_loss = sum_controller_loss / len(self.train_loader)
        # mean_pred_loss = sum_pred_loss / len(self.train_loader)
        # self.writer.add_scalar('train/controller_loss', mean_controller_loss, epoch)
        # self.writer.add_scalar('train/pred_loss', mean_pred_loss, epoch)

    def ControllerLoss(self, prob, sim):
        """
        Loss to train writing controller:
        :param prob: writing probability generated by controller
        :param sim: similarity (between 0 and 1) between better prediction and ground-truth.
        :return: loss
        """
        loss = (prob * sim + (1 - prob) * (1 - sim)).mean()
        return loss

    def _memory_writing(self, saved_memory=False):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        if saved_memory:
            print("Use saved memory")
            return None

        self.mem_n2n.init_memory(self.data_train)
        with torch.no_grad():
            for (index, past, future, _, _, _, _, _, _, _) in tqdm(
                self.train_loader, desc="write memory"
            ):
                self.iterations += 1
                past = Variable(past)
                future = Variable(future)
                past = past.to(self.device)
                future = future.to(self.device)
                _, _ = self.mem_n2n(past=past, future=future)
