import time
import torch
import numpy as np

from GSSFiltering.filtering import DataDriven_Koopman_Kalman_Filter

print_num = 25


def get_default_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def sync_cuda(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_filter_to_device(filter_obj, device):
    """
    将滤波器对象及其包含的神经网络子模块和张量迁移到指定设备。
    """
    device = torch.device(device)

    if hasattr(filter_obj, "to") and callable(filter_obj.to):
        filter_obj.to(device)
        return filter_obj

    if hasattr(filter_obj, "device"):
        filter_obj.device = device

    if hasattr(filter_obj, "kf_net") and hasattr(filter_obj.kf_net, "to"):
        filter_obj.kf_net.to(device)

    tensor_attrs = [
        "init_state",
        "state_post",
        "state_history",
        "last_gain",
        "z_post",
    ]

    for attr in tensor_attrs:
        if hasattr(filter_obj, attr):
            value = getattr(filter_obj, attr)
            if torch.is_tensor(value):
                setattr(filter_obj, attr, value.to(device))

    return filter_obj


class Tester:
    def __init__(
        self,
        filter: DataDriven_Koopman_Kalman_Filter,
        data_path: str,
        model_path: str,
        is_validation: bool = False,
        is_mismatch: bool = False,
        device=None,
        save_xhat: bool = True,
    ):
        """
        适用于 KoopmanNet (DataDriven_Koopman_Kalman_Filter) 的评估器。

        Args:
            filter: DataDriven_Koopman_Kalman_Filter 实例对象。
            data_path: 包含 state.pt 和 obs.pt 的路径。
            model_path: 模型 checkpoint 权重文件路径。
            is_validation: 是否为验证模式。
            is_mismatch: 是否为模型失配模式。
            device: 运算设备 ('cuda', 'cpu' 或 None)。
            save_xhat: 是否将估计结果保存到文件。
        """
        self.device = get_default_device(device)
        self.result_path = "DD-KKF "

        self.filter = move_filter_to_device(filter, self.device)

        # 加载网络权重
        if hasattr(self.filter, "kf_net"):
            loaded = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(loaded, dict) and "kf_net" in loaded:
                self.filter.kf_net.load_state_dict(loaded["kf_net"])
            else:
                self.filter.kf_net.load_state_dict(loaded)
            self.filter.kf_net.to(self.device)

        self.filter = move_filter_to_device(self.filter, self.device)

        self.x_dim = self.filter.x_dim
        self.y_dim = self.filter.y_dim
        self.data_path = data_path
        self.model_path = model_path
        self.is_validation = is_validation
        self.is_mismatch = is_mismatch
        self.save_xhat = save_xhat

        self.loss_fn = torch.nn.MSELoss()

        # 加载测试/验证轨迹数据
        self.data_x = torch.load(
            data_path + "state.pt", map_location=self.device, weights_only=True
        ).to(self.device)
        self.data_y = torch.load(
            data_path + "obs.pt", map_location=self.device, weights_only=True
        ).to(self.device)

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

        assert self.x_dim == self.data_x.shape[1]
        assert self.y_dim == self.data_y.shape[1]
        assert self.seq_len == self.data_y.shape[2]
        assert self.data_num == self.data_y.shape[0]

        x_hat = torch.zeros_like(self.data_x, device=self.device)
        start_time = time.monotonic()
        global_step_times = []

        with torch.no_grad():
            for i in range(self.data_num):
                if i % print_num == 0:
                    mode_str = "Validating" if self.is_validation else "Testing"
                    print(f"{mode_str} {i} / {self.data_num} of {self.model_path}")

                # 每条轨迹重置滤波器状态
                if hasattr(self.filter, "reset"):
                    try:
                        self.filter.reset(clean_history=True)
                    except TypeError:
                        self.filter.reset()
                    self.filter = move_filter_to_device(self.filter, self.device)

                # 初始状态真值注入
                init_state = self.data_x[i, :, 0].reshape((-1, 1)).to(self.device)
                self.filter.state_post = init_state
                if hasattr(self.filter, "state_history"):
                    self.filter.state_history = init_state.detach().clone()

                # 单步逐时刻递归滤波
                sync_cuda(self.device)
                start_seq = time.perf_counter()

                for ii in range(1, self.seq_len):
                    obs_t = self.data_y[i, :, ii].reshape((-1, 1)).to(self.device)
                    self.filter.filtering(obs_t)

                sync_cuda(self.device)
                end_seq = time.perf_counter()

                x_hat[i] = self.filter.state_history[:, -self.seq_len :].to(self.device)

                valid_steps_count = self.seq_len - 1
                if valid_steps_count > 0:
                    avg_step_time = (end_seq - start_seq) / valid_steps_count
                    global_step_times.extend([avg_step_time] * valid_steps_count)

            # 保存估计结果
            if self.save_xhat:
                torch.save(
                    x_hat.detach().cpu(),
                    data_path + self.result_path + "x_hat.pt",
                )

            # 统计 MSE 损失与 dB 指标
            mse = self.loss_fn(self.data_x[:, :, 1:], x_hat[:, :, 1:])
            loss_dB = 10 * torch.log10(mse)
            print(f"loss [dB] = {loss_dB:.4f}")

            # 瞬时误差
            self.loss_instant = torch.zeros(
                self.data_x[:, :, 1:].shape[-1], device=self.device
            )
            for i in range(self.data_x[:, :, 1:].shape[-1]):
                self.loss_instant[i] = self.loss_fn(
                    self.data_x[:, :, i + 1], x_hat[:, :, i + 1]
                )

            self.x_hat = x_hat
            self.loss_instant_dB = 10 * torch.log10(self.loss_instant)

        self.loss = loss_dB
        self.val_nll = None
        self.total_infer_time_s = time.monotonic() - start_time
        self.step_times_s = global_step_times

        # 统计推理时间
        if len(global_step_times) > 0:
            arr = np.array(global_step_times, dtype=np.float64)
            self.step_time_mean_ms = float(arr.mean() * 1000.0)
            self.step_time_std_ms = (
                float(arr.std(ddof=1) * 1000.0) if arr.size > 1 else 0.0
            )
            self.step_time_p95_ms = float(np.percentile(arr, 95) * 1000.0)
        else:
            self.step_time_mean_ms = float("nan")
            self.step_time_std_ms = float("nan")
            self.step_time_p95_ms = float("nan")