"""
Module that contains functions for MANTRA Autoencoder Training
"""
import os
import sys
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

sys.path.append("../../../Mantra")
from models.model_encdec import model_encdec
from torch.autograd import Variable
from tqdm import tqdm


class Trainer:
    def __init__(self, config, dataset_and_loader=None):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """
        # test folder creating
        self.name_test = config.time
        self.folder_test = os.path.join(
            "../models/training_ae",
            config.experiment_tag,
            config.data_dir.split("_middle")[0] + "_" + config.time,
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
        self.settings = {
            "batch_size": config.batch_size,
            "device": config.device,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,
        }
        self.max_epochs = config.max_epochs

        # model
        self.mem_n2n = model_encdec(self.settings)

        # loss
        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        self.criterionLoss = self.criterionLoss.to(self.device)
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

    def draw_track(
        self, past, future, pred=None, index_tracklet=0, num_epoch=0, train=False
    ):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :param train: True or False, indicates whether the sample is in the training or testing set
        :return: None
        """

        fig = plt.figure()
        past = past.cpu().numpy()
        future = future.cpu().numpy()
        plt.plot(past[:, 0], past[:, 1], c="blue", marker="o", markersize=3)
        plt.plot(future[:, 0], future[:, 1], c="green", marker="o", markersize=3)
        if pred is not None:
            pred = pred.cpu().numpy()
            plt.plot(
                pred[:, 0],
                pred[:, 1],
                color="red",
                linewidth=1,
                marker="o",
                markersize=1,
            )
        plt.axis("equal")

        # Save figure in Tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)

        if train:
            self.writer.add_image(
                "Image_train/track" + str(index_tracklet), image.squeeze(0), num_epoch
            )
        else:
            self.writer.add_image(
                "Image_test/track" + str(index_tracklet), image.squeeze(0), num_epoch
            )

        plt.close(fig)

    def fit(self):
        """
        Autoencoder training procedure. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config
        # Training loop
        for epoch in tqdm(
            range(self.start_epoch, config.max_epochs), desc="epochs", leave=True
        ):
            self._train_single_epoch(epoch)
            if epoch % self.config.eval_every == 0:
                self.evaluate(self.test_loader, epoch)
                torch.save(
                    self.mem_n2n, os.path.join(self.folder_test, f"model-{epoch}")
                )
        # torch.save(self.mem_n2n, os.path.join(self.folder_test, 'model'))

    def evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """
        # Loop over samples
        loss_sum = 0
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
        ) in enumerate(tqdm(loader, desc="samples", leave=False)):
            past = Variable(past)
            future = Variable(future)
            past = past.to(self.device)
            future = future.to(self.device)
            output = self.mem_n2n(past, future)
            loss = self.criterionLoss(output, future)
            loss_sum += loss.detach()
        loss_mean = loss_sum / len(loader)
        self.writer.add_scalar("eval/loss", loss_mean, epoch)

    def _train_single_epoch(self, epoch):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        sum_loss = 0
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
        ) in enumerate(tqdm(self.train_loader, desc="batches", leave=False)):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            past = past.to(self.device)
            future = future.to(self.device)
            self.opt.zero_grad()

            # Get prediction and compute loss
            output = self.mem_n2n(past, future)
            loss = self.criterionLoss(output, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            # Tensorboard summary: loss
            sum_loss += loss.detach()
        mean_loss = sum_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", mean_loss, epoch)


def log_results(writer, dict_metrics_test, epoch, train):
    """
    Serialize results
    :param dict_metrics_test: dictionary with test metrics
    :param dict_metrics_train: dictionary with train metrics
    :param epoch: epoch index (default: 0)
    :return: None
    """

    mode = "train" if train else "eval"
    for metric, metric_value in dict_metrics_test.items():
        writer.add_scalar(f"{mode}/{metric}", metric_value, epoch)
