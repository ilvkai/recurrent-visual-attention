import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import baseline_network
from modules import glimpse_network, core_network
from modules import action_network, location_network, speed_course_network, scale_network
from torchvision.models import resnet50


class RNN_network(nn.Module):
    def __init__(self,
                 std,
                 hidden_size):
        super(RNN_network, self).__init__()
        self.std = std
        hidden_size =2048

        self.speed_course_network = speed_course_network(2, 256)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator_final = location_network(hidden_size, 2, std)
        self.feature_extractor = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])

    def forward(self, x, speeds, courses, l_t_prev, h_t_prev, frame_index, last=False):
        x = x[: ,:, frame_index, :, :]
        x= self.feature_extractor(x).squeeze()
        speed = speeds[:, frame_index].view(x.shape[0], -1)
        course = courses[:, frame_index].view(x.shape[0], -1)
        speed_course = torch.cat((speed, course), 1)
        speed_course = self.speed_course_network(speed_course)


        # g_t = torch.cat((g_t, speed_course), 1)
        h_t = self.rnn(x, h_t_prev)

        # if last:
        #     log_probas = self.classifier(h_t)
        #     return h_t, l_t, b_t, log_probas, log_pi
        if last:
            l_t_final, l_t_final_noise = self.locator_final(h_t)
            return h_t, l_t_final

        return h_t