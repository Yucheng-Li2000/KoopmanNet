import torch
from GSSFiltering.dnn import OnlineKoopmanKNet
from GSSFiltering.model import GSSModel


def get_default_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def init_hidden_safe(net, device):
    if hasattr(net, "initialize_hidden"):
        try:
            net.initialize_hidden(device=device)
        except TypeError:
            net.initialize_hidden()


class DataDriven_Koopman_Kalman_Filter:
    def __init__(self, GSSModel: GSSModel, n_features: int, device=None):
        # 初始设备解析
        self._device = get_default_device(device)

        self.n_features = n_features
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel
        self.last_gain = None

        # 初始化网络并移至指定设备
        self.kf_net = OnlineKoopmanKNet(
            self.x_dim,
            self.y_dim,
            n_features
        ).to(self._device)

        self.state_history = None
        self.reset(clean_history=True)

    @property
    def device(self):
        """动态获取模型当前所在的实际设备"""
        try:
            return next(self.kf_net.parameters()).device
        except (StopIteration, AttributeError):
            return self._device

    @device.setter
    def device(self, val):
        """允许外部直接给 filter_obj.device = device 赋值并同步转移模型"""
        self._device = torch.device(val)
        if hasattr(self, 'kf_net') and hasattr(self.kf_net, 'to'):
            self.kf_net.to(self._device)

    def to(self, device):
        """
        支持实例调用 .to(device)，同步转移模型和内部所有缓存张量
        """
        self._device = torch.device(device)
        self.kf_net = self.kf_net.to(self._device)
        
        if hasattr(self, 'state_post') and self.state_post is not None:
            self.state_post = self.state_post.to(self._device)
        if hasattr(self, 'state_history') and self.state_history is not None:
            self.state_history = self.state_history.to(self._device)
        if hasattr(self, 'last_gain') and self.last_gain is not None:
            self.last_gain = self.last_gain.to(self._device)
        return self

    def reset(self, clean_history=False):
        curr_device = self.device
        init_hidden_safe(self.kf_net, curr_device)

        self.dnn_first = True
        # 确保初始状态直接放到模型所在设备上
        self.state_post = self.GSSModel.init_state.detach().clone().to(curr_device)
        self.last_gain = None

        if clean_history:
            self.state_history = self.GSSModel.init_state.detach().clone().to(curr_device)

    def filtering(self, y_obs: torch.Tensor):
        # 获取模型实际所在设备，统一所有张量
        curr_device = self.device

        y_obs = y_obs.to(curr_device)
        self.state_post = self.state_post.to(curr_device)

        # 1. 编码器前向传播 (此时数据和网络必在同一设备)
        self.z_post = self.kf_net.encoder(self.state_post.T).reshape(-1, 1)

        # 2. KalmanNet 步骤
        G, z_pred, y_pred = self.kf_net(self.z_post, y_obs)

        self.last_gain = G
        self.z_post = z_pred + G @ (y_obs - y_pred)

        # 确保 B 矩阵如果在模型内部是普通张量时也在同设备
        B = self.kf_net.B.to(curr_device) if hasattr(self.kf_net, 'B') else None
        x_post = B @ self.z_post if B is not None else self.z_post

        # 3. 状态更新与历史追加
        self.state_post = x_post.detach().clone()

        if self.state_history is None:
            self.state_history = x_post.detach().clone()
        else:
            self.state_history = torch.cat(
                (self.state_history.to(curr_device), x_post), 
                dim=1
            )

        return x_post