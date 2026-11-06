import os
import time
import configparser
import torch
import torch.optim as optim

from GSSFiltering.filtering import DataDriven_Koopman_Kalman_Filter
from GSSFiltering.tester import Tester

config = configparser.ConfigParser()
config.read('./config.ini')

if not os.path.exists('./.model_saved'):
    os.mkdir('./.model_saved')

print_num = 25
save_num = int(config['Train']['valid_period'])
train_iter = int(config['Train']['train_iter'])
lr_koopman = float(config['Train.Koopman']['learning_rate'])
wd_kalman = float(config['Train.Kalman']['weight_decay'])


class Trainer:
    def __init__(
        self,
        dnn: DataDriven_Koopman_Kalman_Filter,
        data_path: str,
        save_path: str,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.dnn = dnn
        self.data_path = data_path
        self.save_path = save_path
        self.save_num = save_num

        self.x_dim = self.dnn.x_dim
        self.y_dim = self.dnn.y_dim
        self.loss_best = 1e4

        self.train_forward_time_ms = []
        self.last_train_forward_ms = float('nan')

        # ===================== 加载训练数据 =====================
        self.data_x = torch.load(
            data_path + 'state.pt', map_location=self.device, weights_only=True
        ).to(self.device)
        self.data_y = torch.load(
            data_path + 'obs.pt', map_location=self.device, weights_only=True
        ).to(self.device)

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

        assert self.x_dim == self.data_x.shape[1]
        assert self.y_dim == self.data_y.shape[1]
        assert self.seq_len == self.data_y.shape[2]
        assert self.data_num == self.data_y.shape[0]

        # ===================== 迁移模型至设备 =====================
        self._move_dnn_to_device()

        # ===================== 优化器与损失函数 =====================
        self.loss_fn = torch.nn.MSELoss()
        params = list(self.dnn.kf_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr_koopman, weight_decay=wd_kalman)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.batch_size = int(config['Train']['batch_size'])
        self.train_count = 0
        self.data_idx = 0

    def _move_dnn_to_device(self):
        if hasattr(self.dnn, "to") and callable(self.dnn.to):
            self.dnn.to(self.device)
            return

        if hasattr(self.dnn, "kf_net"):
            self.dnn.kf_net.to(self.device)

        tensor_attrs = [
            "init_state",
            "state_post",
            "state_history",
            "last_gain",
            "z_post",
        ]
        for attr in tensor_attrs:
            if hasattr(self.dnn, attr):
                val = getattr(self.dnn, attr)
                if torch.is_tensor(val):
                    setattr(self.dnn, attr, val.to(self.device))

    def _sync_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _get_batch(self):
        if self.data_idx + self.batch_size >= self.data_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(self.data_x.shape[0], device=self.device)
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]

        batch_x = self.data_x[self.data_idx : self.data_idx + self.batch_size].to(self.device)
        batch_y = self.data_y[self.data_idx : self.data_idx + self.batch_size].to(self.device)
        return batch_x, batch_y

    def train_batch(self):
        """单批次训练入口"""
        forward_ms = self.train_batch_joint()
        self.last_train_forward_ms = float(forward_ms)
        self.train_forward_time_ms.append(self.last_train_forward_ms)
        return self.last_train_forward_ms

    def train_batch_joint(self):
        self.optimizer.zero_grad(set_to_none=True)
        batch_x, batch_y = self._get_batch()

        self._sync_cuda()
        t_fwd0 = time.perf_counter()

        x_hat = torch.zeros_like(batch_x, device=self.device)
        rec_loss = torch.tensor(0.0, device=self.device)
        obs_loss = torch.tensor(0.0, device=self.device)
        dyna_loss = torch.tensor(0.0, device=self.device)

        for i in range(self.batch_size):
            self.dnn.state_post = batch_x[i, :, 0].reshape((-1, 1)).to(self.device)

            for ii in range(1, self.seq_len):
                y_obs = batch_y[i, :, ii].reshape((-1, 1)).to(self.device)
                x_true = batch_x[i, :, ii].reshape((-1, 1)).to(self.device)

                # 1. 编码与状态重构损失 (Reconstruction Loss)
                z_true = self.dnn.kf_net.encoder(x_true.T).reshape(-1, 1)
                x_rec = self.dnn.kf_net.B @ z_true
                rec_loss += self.loss_fn(x_true, x_rec)

                # 2. 观测映射损失 (Observation Loss)
                y_pred = (self.dnn.kf_net.C.T @ z_true).view(-1, 1)
                obs_loss += self.loss_fn(y_obs, y_pred)

                # 3. 潜空间动力学推进损失 (Linear Dynamics Loss)
                if ii < self.seq_len - 1:
                    x_next_true = batch_x[i, :, ii + 1].reshape((-1, 1)).to(self.device)
                    z_pred = (self.dnn.kf_net.U @ z_true).view(-1, 1)
                    z_next_true = self.dnn.kf_net.encoder(x_next_true.T).reshape(-1, 1)
                    dyna_loss += self.loss_fn(z_pred, z_next_true)

                # 4. 执行 Koopman Kalman 滤波前向迭代
                self.dnn.filtering(y_obs)

            x_hat[i] = self.dnn.state_history[:, -self.seq_len:].to(self.device)

            if hasattr(self.dnn, "reset"):
                self.dnn.reset(clean_history=False)
                self._move_dnn_to_device()

        denom = float(self.batch_size * (self.seq_len - 1))
        rec_loss = rec_loss / denom
        obs_loss = obs_loss / denom
        dyna_loss = dyna_loss / denom

        state_loss = self.loss_fn(batch_x[:, :, 1:], x_hat[:, :, 1:])

        loss = state_loss + rec_loss + obs_loss + dyna_loss

        self._sync_cuda()
        forward_ms = (time.perf_counter() - t_fwd0) * 1000.0

        if self.train_count % print_num == 1:
            print(
                f'[Model {self.save_path}] [Train {self.train_count}] '
                f'loss [dB] = {(10 * torch.log10(loss)).item():.4f}'
            )

        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self.train_count += 1
            self.data_idx += self.batch_size
            return forward_ms

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)
        self.optimizer.step()

        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr > 5e-4:
            self.scheduler.step()

        self.train_count += 1
        self.data_idx += self.batch_size

        # 定期保存检查点
        if self.train_count % save_num == 0:
            save_dict = {
                'kf_net': self.dnn.kf_net.state_dict(),
            }
            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt'
            )

        return forward_ms

    def validate(self, tester: Tester):
        """验证并保存最佳模型"""
        if tester.loss.item() < self.loss_best:
            save_dict = {
                'kf_net': tester.filter.kf_net.state_dict()
            }
            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_best.pt'
            )
            print(
                f'Save best model at {self.save_path} & train {self.train_count} '
                f'& loss [dB] = {tester.loss:.4f}'
            )
            self.loss_best = tester.loss.item()

        self.valid_loss = tester.loss.item()