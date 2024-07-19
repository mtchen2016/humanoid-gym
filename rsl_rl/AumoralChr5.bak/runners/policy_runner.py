import os
import time
import copy
from collections import deque
import statistics
from termcolor import cprint

from torch.utils.tensorboard import SummaryWriter
from umoralChr5.utils.utils import export_policy_as_jit, export_policy_as_onnx, export_cnn_as_onnx
import torch

from AumoralChr5.algorithms import PPO
from AumoralChr5.modules import ActorCritic, ActorCriticRecurrent
from AumoralChr5.env import VecEnv


class AumoralChr5PolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.encoder_cfg = train_cfg["Encoder"]

        self.encoder_mlp = train_cfg["Encoder"]['encoder_mlp_units']
        self.device = device
        self.env = env
        self.HistoryLen = train_cfg["Encoder"]['HistoryLen']

        ### add image shape ###
        self.MLP_type = train_cfg["Encoder"]['encoder_type']
        self.camera_dim = train_cfg["Encoder"]['camera_dim']


        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg,
                                                       **self.encoder_cfg).to(self.device)


        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic,
                                  device=self.device,
                                  **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_encoder_input = self.env.num_obs * self.HistoryLen
        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions],
                              [self.num_encoder_input], self.camera_dim)

        # Log
        self.log_dir = log_dir
        self.nn_dir = os.path.join(self.log_dir, 'stage1_nn')
        self.tb_dir = os.path.join(self.log_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_dict)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict)

            mean_value_loss, mean_surrogate_loss, mean_vel_loss, mean_mass_loss, mean_cnn_map_height_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.nn_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.nn_dir, 'model_{}.pt'.format(it)))
                self.save(os.path.join(self.nn_dir, 'last.pt'))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.nn_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Vel/vel_loss', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Vel/morph_loss', locs['mean_mass_loss'], locs['it'])

        self.writer.add_scalar('Cnn/map_height_loss', locs['mean_cnn_map_height_loss'], locs['it'])

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'actor_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)


    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['actor_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']

        # export policy as a jit module (used to run it from C++)
        if self.cfg['export_policy'] == "pt":
            cprint('Exporting policy to jit module(C++)', 'green', attrs=['bold'])
            jit_save_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'pt_s1')
            export_policy_as_jit(self.alg.actor_critic.actor, jit_save_path, 'actor.pt')
            export_policy_as_jit(self.alg.actor_critic.dm_encoder, jit_save_path, 'encoder.pt')

            print(f"actor: {self.alg.actor_critic.actor}")

        if self.cfg['export_policy'] == "onnx":
            cprint('Exporting policy to onnx module(C++)', 'red', attrs=['bold'])
            onnx_save_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'onnx_s1')
            # calculate input size and save as onnx model
            print()
            export_policy_as_onnx(self.alg.actor_critic.actor,
                                  self.alg.actor_critic.num_actor_input,
                                  onnx_save_path,
                                  'actor.onnx',
                                  input_names=["observation"],
                                  output_names=["action"],
                                 )
            print()
            export_policy_as_onnx(self.alg.actor_critic.dm_encoder,
                                  self.alg.actor_critic.num_encoder_input,
                                  onnx_save_path,
                                  'est_encoder.onnx',
                                  input_names=["obs_history"],
                                  output_names=["est_feature"],
                                  )

            print()
            export_policy_as_onnx(self.alg.actor_critic.height_encoder,
                                  self.alg.actor_critic.num_height_input,
                                  onnx_save_path,
                                  'height_encoder.onnx',
                                  input_names=["height_point"],
                                  output_names=["height_feature"],
                                  )
            # print()
            # input_shape = (1, 2, 16, 16)
            # export_cnn_as_onnx(self.alg.actor_critic.cnn_encoder,
            #                       input_shape,
            #                       onnx_save_path,
            #                       'cnn.onnx',
            #                        input_names=["imgae_shape"],
            #                        output_names=["imgae_feature"],
            #                    )

            print(f"exported onnx actor: {self.alg.actor_critic.actor}")
            print(f"exported onnx encoder: {self.alg.actor_critic.dm_encoder}")
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.evaluate

