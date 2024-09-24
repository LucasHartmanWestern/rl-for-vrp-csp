import copy
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Normal  # Use Normal distribution for continuous actions


class TrainerConfig:
    # optimization parameters
    max_epochs = 1000
    batch_size = 128
    learning_rate = 5e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 0.5
    weight_decay = 0.1  # only applied on matmul weights
    # checkpoint settings
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, critic_model, config):
        self.model = model
        self.critic_model = critic_model
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = self.raw_model.configure_optimizers(config, config.learning_rate)

        self.raw_critic_model = self.critic_model.module if hasattr(self.critic_model, "module") else self.critic_model
        self.critic_optimizer = self.raw_critic_model.configure_optimizers(config, config.learning_rate * 10)

    def train(self, dataset, train_critic=True):
        model, critic_model, config = self.raw_model, self.raw_critic_model, self.config
        target_model = copy.deepcopy(model)
        target_model.train(False)

        def run_epoch():
            model.train(True)
            critic_model.train(True)
            if self.config.mode == "offline":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            elif self.config.mode == "online":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=dataset.__len__(),
                                    num_workers=config.num_workers)
            else:
                raise NotImplementedError       

            loss_info = 0
            pbar = tqdm(enumerate(loader), total=len(loader))
            # todo: check these inputs

            for it, (s, o, a, r, ava, v, rtg, ret, adv, t, pre_a, next_s, next_rtg, done) in pbar:
                
                # place data on the correct device
                s = s.to(self.device)
                o = o.to(self.device)
                a = a.to(self.device).float()
                r = r.to(self.device)
                ava = ava.to(self.device)
                v = v.to(self.device)
                rtg = rtg.to(self.device)
                ret = ret.to(self.device)
                adv = adv.to(self.device)
                t = t.to(self.device)
                pre_a = pre_a.to(self.device)
                next_s = next_s.to(self.device)
                next_rtg = next_rtg.to(self.device)
                done = done.to(self.device)
            
                # update actor
                with torch.set_grad_enabled(True):
                    # Unpack the tuple returned by the actor model
                    action_mean, action_std = model(o, pre_a, rtg, t)
                    if self.config.mode == "offline":
            
                        # Compute the action loss using the action mean
                        loss = F.mse_loss(action_mean.reshape(-1, action_mean.size(-1)), a.reshape(-1, a.size(-1)))
                        entropy_info = 0.
                        ratio_info = 0.
                        confidence_info = 0.
                    elif self.config.mode == "online":
                        adv = adv.reshape(-1, adv.size(-1))
            
                        # # Here we assume a Gaussian distribution to sample actions based on the predicted mean and std
                        # print("action_mean:", action_mean)
                        # print("action_std:", action_std)
                        if torch.isnan(action_mean).any():
                            epsilon = 1e-6
                            action_mean = torch.nan_to_num(action_mean, nan=0.0) + epsilon
                        if torch.isnan(action_std).any():
                            action_std = torch.nan_to_num(action_std, nan=0.0) + epsilon
                        normal_dist = torch.distributions.Normal(action_mean, action_std) 
                        log_a = normal_dist.log_prob(a).sum(-1, keepdim=True)
            
                        old_action_mean, old_action_std = target_model(o, pre_a, rtg, t)
                        old_action_mean = old_action_mean.detach()
                        old_action_std = old_action_std.detach()

                        epsilon = 1e-6  # Small value to avoid zero in the standard deviation
                        old_action_std = torch.nan_to_num(old_action_std, nan=epsilon)
                        old_action_std = torch.clamp(old_action_std, min=epsilon)  # Ensure std is positive
                        old_action_mean = torch.nan_to_num(old_action_mean, nan=0.0)
                     
                        old_normal_dist = torch.distributions.Normal(old_action_mean, old_action_std)
                        old_log_a = old_normal_dist.log_prob(a).sum(-1, keepdim=True)
            
                        imp_weights = torch.exp(log_a - old_log_a)
                        actor_loss_ori = imp_weights * adv
                        actor_loss_clip = torch.clamp(imp_weights, 1.0 - 0.2, 1.0 + 0.2) * adv
                        actor_loss = -torch.min(actor_loss_ori, actor_loss_clip)
            
                        act_entropy = normal_dist.entropy().mean()
                        loss = actor_loss - 0.01 * act_entropy
            
                        entropy_info = act_entropy.mean().item()
                        ratio_info = imp_weights.mean().item()
                        confidence_info = torch.exp(log_a).mean().item()
                    else:
                        raise NotImplementedError
            
                    loss = loss.mean()
                    loss_info = loss.item()
            
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
            
                # update critic
                critic_loss_info = 0.
                if train_critic:
                    with torch.set_grad_enabled(True):
                        v_value = critic_model(s, pre_a, rtg, t)
                        v_clip = v + (v_value - v).clamp(-0.2, 0.2)
                        critic_loss_ori = F.smooth_l1_loss(v_value.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss_clip = F.smooth_l1_loss(v_clip.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss = torch.max(critic_loss_ori, critic_loss_clip)
            
                        critic_loss_info = critic_loss.mean().item()
            
                    critic_model.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_model.parameters(), config.grad_norm_clip)
                    self.critic_optimizer.step()


                # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")
            return loss_info, critic_loss_info, entropy_info, ratio_info, confidence_info

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0., 0.
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = run_epoch()
        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence
