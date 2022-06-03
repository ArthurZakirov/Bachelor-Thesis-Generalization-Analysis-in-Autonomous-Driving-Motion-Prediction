import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

sys.path.append('../../../Mantra/models')
from model_utils import get_tolerance_rate

class model_controllerMem(nn.Module):
    """
    Memory Network model with learnable writing controller.
    """

    def __init__(self, settings, model_pretrained):
        super(model_controllerMem, self).__init__()
        self.name_model = 'writing_controller'
        # parameters
        self.device = settings["device"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.num_prediction = settings["num_prediction"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]


        # similarity criterion
        self.weight_read = []
        self.index_max = []
        self.similarity = nn.CosineSimilarity(dim=1)

        # Memory
        self.memory_past = torch.Tensor().to(self.device)
        self.memory_fut = torch.Tensor().to(self.device)
        self.memory_count = []

        # layers
        self.model_ae = model_pretrained

        ########################################################################################
        # diesen block braucht man nicht, aber man muss ihn lassen um das model laden zu können
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut
        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output
        ########################################################################################

        # activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.linear_controller = torch.nn.Linear(1, 1)

    def init_memory(self, data_train):
        """
        Initialization: write samples in memory.
        :param data_train: dataset
        :return: None
        """
        self.memory_past = torch.Tensor().to(self.device)
        self.memory_fut = torch.Tensor().to(self.device)

        for i in range(self.num_prediction):

            # random element from train dataset to be added in memory
            j = random.randint(0, len(data_train)-1)
            past = data_train[j][1].unsqueeze(0).to(self.device)
            future = data_train[j][2].unsqueeze(0).to(self.device)

            # plt.plot(past[0, :, 0], past[0, :, 1], label='past')
            # plt.plot(future[0, :, 0], future[0, :, 1], label='future')

            # past encoding
            past = torch.transpose(past, 1, 2)
            story_embed = self.relu(self.model_ae.conv_past(past))
            story_embed = torch.transpose(story_embed, 1, 2)
            output_past, state_past = self.model_ae.encoder_past(story_embed)

            # future encoding
            future = torch.transpose(future, 1, 2)
            future_embed = self.relu(self.model_ae.conv_fut(future))
            future_embed = torch.transpose(future_embed, 1, 2)
            output_fut, state_fut = self.encoder_fut(future_embed)

            # insert in memory
            self.memory_past = torch.cat((self.memory_past, state_past.squeeze(0)), 0)
            self.memory_fut = torch.cat((self.memory_fut, state_fut.squeeze(0)), 0)

            # plot decoding of memory sample
            # mem_past_i = state_past.squeeze(0).squeeze(0)
            # mem_fut_i = state_fut.squeeze(0).squeeze(0)
            # zero_padding = torch.zeros(1, 1, 96).to(self.device)
            # present = torch.zeros(1, 2).to(self.device)
            # prediction_single = torch.Tensor().to(self.device)
            # info_total = torch.cat((mem_past_i, mem_fut_i), 0)
            # input_dec = info_total.unsqueeze(0).unsqueeze(0)
            # state_dec = zero_padding
            # for i in range(self.future_len):
            #     output_decoder, state_dec = self.model_ae.decoder(input_dec, state_dec)
            #     displacement_next = self.model_ae.FC_output(output_decoder)
            #     coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            #     prediction_single = torch.cat((prediction_single, coords_next), 1)
            #     present = coords_next
            #     input_dec = zero_padding
            # pred = prediction_single
            # plt.plot(pred[0, :, 0].detach().numpy(), pred[0, :, 1].detach().numpy(), label='pred')
            # plt.legend(loc='best')
            # plt.axis('equal')
            # plt.show()


    def check_memory(self, index):
        """
        Method to generate a future track from past-future feature read from an index location of the memory.
        :param index: index of the memory
        :return: predicted future
        """
        mem_past_i = self.memory_past[index]
        mem_fut_i = self.memory_fut[index]
        zero_padding = torch.zeros(1, 1, 96).to(self.device)
        present = torch.zeros(1, 2).to(self.device)
        prediction_single = torch.Tensor().to(self.device)
        info_total = torch.cat((mem_past_i, mem_fut_i), 0)
        input_dec = info_total.unsqueeze(0).unsqueeze(0)
        state_dec = zero_padding
        for i in range(self.future_len):
            output_decoder, state_dec = self.model_ae.decoder(input_dec, state_dec)
            displacement_next = self.model_ae.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction_single = torch.cat((prediction_single, coords_next), 1)
            present = coords_next
            input_dec = zero_padding
        return prediction_single

    def predict(self, past):
        """
        Forward pass.
        Train phase: training writing controller based on reconstruction error of the future.
        Test phase: Predicts future trajectory based on past trajectory and the future feature read from the memory.
        :param past: past trajectory
        :param future: future trajectory (in test phase)
        :return: predicted future (test phase), writing probability and tolerance rate (train phase)
        """
        num_prediction = min(self.memory_past.shape[0], self.num_prediction)
        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 2).to(self.device)
        prediction = torch.Tensor().to(self.device)
        present_temp = past[:, -1].unsqueeze(1)

        # past temporal encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.model_ae.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.model_ae.encoder_past(story_embed)

        # Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1) # [bs, M, dim]
        state_normalized = F.normalize(state_past.squeeze(0), p=2, dim=1) # [bs, dim]
        weight_read = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1) #[bs, ] #/ (torch.norm(past_normalized, dim=-1) * torch.norm(state_normalized, dim=-1))
        index_max = torch.sort(weight_read, descending=True)[1].cpu()

        for i_track in range(num_prediction):
            present = present_temp
            prediction_single = torch.Tensor().to(self.device)
            ind = index_max[:, i_track]
            info_future = self.memory_fut[ind]
            info_total = torch.cat((state_past, info_future.unsqueeze(0)), 2)
            input_dec = info_total
            state_dec = zero_padding
            for i in range(self.future_len):
                output_decoder, state_dec = self.model_ae.decoder(input_dec, state_dec)
                displacement_next = self.model_ae.FC_output(output_decoder)
                coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                prediction_single = torch.cat((prediction_single, coords_next), 1)
                present = coords_next
                input_dec = zero_padding
            prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)
        return prediction, state_past

    def write_in_memory(self, prediction, future, state_past):
        tolerance_rate = get_tolerance_rate(prediction, future)
        tolerance_rate = tolerance_rate.to(self.device)

        # controller
        #writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))
        writing_prob = 1 - tolerance_rate
        # future encoding
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.model_ae.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)

        bs = tolerance_rate.shape[0]
        #index_writing = np.where(writing_prob.cpu() > self.min_writing_threshold)[0]
        self.min_tolerance_rate = 0.3 # 0.3 used in training, 0.7 used for proving the point, 1.0 for online writing
        index_writing = np.where(tolerance_rate.cpu() < self.min_tolerance_rate)[0]
        try:
            past_to_write = state_past.squeeze()[index_writing]
            future_to_write = state_fut.squeeze()[index_writing]
            self.memory_past = torch.cat((self.memory_past, past_to_write), dim=0)
            self.memory_fut = torch.cat((self.memory_fut, future_to_write), dim=0)
        except:
            past_to_write = state_past.squeeze(1)[index_writing]
            future_to_write = state_fut.squeeze(1)[index_writing]
            self.memory_past = torch.cat((self.memory_past, past_to_write), dim=0)
            self.memory_fut = torch.cat((self.memory_fut, future_to_write), dim=0)

        return writing_prob, tolerance_rate

    def forward(self, past, future=None):
        prediction, state_past = self.predict(past)
        if future is not None:
            writing_prob, tolerance_rate = self.write_in_memory(prediction, future, state_past)
            return writing_prob, tolerance_rate
        else:
            return prediction

