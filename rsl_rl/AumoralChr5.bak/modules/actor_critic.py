import numpy as np
from termcolor import cprint

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, num_classes):
        super(CNNEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.encoder2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        f1 = self.encoder1(x)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = f3.view(f3.size(0), -1)
        x = self.fc(f4)
        return x


class DmEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """

        return self.encoder(dm)


class HeightEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """
        return self.encoder(dm)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        # ---- Priv Info ----
        self.MLP_type = kwargs['encoder_type']
        self.MLP_subtype = kwargs['encoder_sub_type']

        self.num_encoder_input = num_obs * kwargs['HistoryLen']

        if self.MLP_type == "Aumoral":
            self.dm_encoder = DmEncoder(self.num_encoder_input, kwargs['encoder_mlp_units'])
        elif self.MLP_type == "AumoralFeature":
            self.num_height_input = 187
            self.dm_encoder = DmEncoder(self.num_encoder_input, kwargs['encoder_mlp_units'])
            self.height_encoder = HeightEncoder(self.num_height_input, kwargs['heigh_mlp_units'])
            cprint(f"height_encoder CNN: {self.height_encoder}", 'red', attrs=['bold'])
        else:
            pass
        if self.MLP_type == "Aumoral":
            self.num_actor_input = num_obs + kwargs['encoder_mlp_units'][2]
            self.num_critic_input = num_obs + kwargs['encoder_mlp_units'][2] + 187 + kwargs['priv_info_dim']  ##### 45 + 3 + 187
        elif self.MLP_type == "AumoralFeature":
            self.num_actor_input = num_obs + kwargs['encoder_mlp_units'][2] + kwargs['heigh_mlp_units'][2]
            if self.MLP_subtype == 0:
                self.num_critic_input = num_obs + kwargs['encoder_mlp_units'][2]+ kwargs['heigh_mlp_units'][2]+ kwargs['priv_info_dim']  ##### 45 + 3 + 187
            elif self.MLP_subtype == 1:
                self.num_critic_input = num_obs + kwargs['encoder_mlp_units'][2]+ 187+ kwargs['priv_info_dim']  ##### 45 + 3 + 187

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs_dict, **kwargs):
        # self.update_distribution(observations)
        mean, std, _, e = self._actor_critic(obs_dict)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        _, _, value, extrin = self._actor_critic(obs_dict)
        return value, extrin


    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        obs_vel = obs_dict['privileged_info'][:, 0:3]
        # foot_height, contact, 
        obs_foot = obs_dict['privileged_info'][:,3:7]
        # 11 * 17 = 187
        obs_hight = obs_dict['privileged_info'][:, 7:194]
        # 2 force
        obs_push = obs_dict['privileged_info'][:, 194:196]
        # see legged2_robot_terrain
        # (0:6) whole mass
        obs_morph = obs_dict['privileged_info'][:, 196:202]
        # temp is (6:20), whole motor info
        obs_kp_kd = obs_dict['privileged_info'][:, 202:216]
        # (20:21)
        obs_friction = obs_dict['privileged_info'][:, 216:217]
        # (21:23)
        obs_center = obs_dict['privileged_info'][:, 217:219]

        extrin_encoder = self.dm_encoder(obs_dict['proprio_hist'])

        if self.MLP_type == "Aumoral":
            actor_obs = torch.cat([extrin_encoder[:, 0:3], obs_dict['obs'], extrin_encoder[:, 3:]], dim=-1)  ## 3 + 39 + 6 = 48
            critic_obs = torch.cat([obs_vel, obs_dict['obs'], obs_morph, obs_foot, obs_hight,## 48+4+187=239
                                    obs_push, obs_friction, obs_center, obs_kp_kd], dim=-1)  # 2+1+2+14=19
            extrin = [extrin_encoder, None]

        elif self.MLP_type == "AumoralFeature":
            extrin_height = self.height_encoder(obs_hight)
            actor_obs = torch.cat([extrin_encoder[:, 0:3], obs_dict['obs'], extrin_encoder[:, 3:], extrin_height],
                                  dim=-1)  ## 3 + 45 + 4 + 16

            if self.MLP_subtype == 0:
                critic_obs = torch.cat([obs_vel, obs_dict['obs'], obs_morph, obs_foot, extrin_height,
                                        obs_push, obs_friction, obs_center, obs_kp_kd], dim=-1)  ## 45+3+187 = 235

            elif self.MLP_subtype == 1:
                critic_obs = torch.cat([obs_vel, obs_dict['obs'], obs_morph, obs_foot, obs_hight,
                                        obs_push, obs_friction, obs_center, obs_kp_kd], dim=-1)  ## 45+3+187 = 235
            extrin = [extrin_encoder, extrin_height]

        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std


        return mu, mu * 0 + sigma, value, extrin


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


