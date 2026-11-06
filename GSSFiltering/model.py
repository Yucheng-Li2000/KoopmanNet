import os
import math
import random
import configparser
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

config = configparser.ConfigParser()
config.read('./config.ini')

if not os.path.exists('./.data'):
    os.mkdir('./.data')


def get_default_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GSSModel():
    def __init__(self, device=None):
        self.device = get_default_device(device)
        self.cov_q = None
        self.cov_r = None
        self.x_dim = None
        self.y_dim = None

    def to(self, device):
        self.device = get_default_device(device)
        for name in ["cov_q", "cov_r", "init_state", "init_cov", "F", "H", "H_true",
                     "q2", "r2", "axis"]:
            if hasattr(self, name):
                value = getattr(self, name)
                if torch.is_tensor(value):
                    setattr(self, name, value.to(self.device))
        return self

    def generate_data(self):
        raise NotImplementedError()

    def f(self, current_state):
        raise NotImplementedError()

    def g(self, current_state):
        raise NotImplementedError()

    def Jacobian_f(self, x):
        raise NotImplementedError()

    def Jacobian_g(self, x):
        raise NotImplementedError()

    def sample_noise(self, cov, dim, device, dtype):
        """
        统一噪声采样函数：支持 'gaussian'、'gmm'、'laplace'
        三种噪声的总体协方差均严格等于传入的 cov，确保公平对比
        """
        noise_type = getattr(self, "noise_type", "laplace").lower()
        mean = torch.zeros(dim, device=device, dtype=dtype)

        # 1. 标准高斯噪声
        if noise_type in ["gaussian", "gauss"]:
            return MultivariateNormal(mean, covariance_matrix=cov).sample().reshape((-1, 1))

        # 2. 混合高斯噪声 (GMM)
        elif noise_type in ["gmm", "mixture"]:
            p = getattr(self, "gmm_p", 0.1)
            k = getattr(self, "gmm_ratio", 10.0)
            s1_sq = 1.0 / (1.0 - p + p * k)
            s2_sq = k * s1_sq

            if torch.rand(1, device=device).item() < p:
                comp_cov = s2_sq * cov
            else:
                comp_cov = s1_sq * cov
            return MultivariateNormal(mean, covariance_matrix=comp_cov).sample().reshape((-1, 1))

        # 3. 拉普拉斯噪声
        elif noise_type in ["laplace", "laplacian"]:
            try:
                L = torch.linalg.cholesky(cov)
            except Exception:
                jitter = 1e-8 * torch.eye(dim, device=device, dtype=dtype)
                L = torch.linalg.cholesky(cov + jitter)

            b = 1.0 / math.sqrt(2.0)
            laplace_dist = torch.distributions.Laplace(
                loc=torch.zeros(dim, device=device, dtype=dtype),
                scale=torch.full((dim,), b, device=device, dtype=dtype)
            )
            z = laplace_dist.sample().reshape((-1, 1))
            return L @ z

        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}. Choose from ['gaussian', 'gmm', 'laplace'].")

    def next_state(self, current_state):
        current_state = current_state.to(getattr(self, "device", current_state.device))
        cov_q = self.cov_q.to(device=current_state.device, dtype=current_state.dtype)
        noise = self.sample_noise(cov_q, self.x_dim, current_state.device, current_state.dtype)
        return self.f(current_state) + noise

    def observe(self, current_state):
        current_state = current_state.to(getattr(self, "device", current_state.device))
        cov_r = self.cov_r.to(device=current_state.device, dtype=current_state.dtype)
        noise = self.sample_noise(cov_r, self.y_dim, current_state.device, current_state.dtype)
        return self.g(current_state) + noise


# ==============================================================================
# 1. 正弦匹配模型 (SyntheticSinusoidalModel)
# ==============================================================================
class SyntheticSinusoidalModel(GSSModel):
    def __init__(self, mode='train', device=None):
        super().__init__(device=device)
        self.config = config['SyntheticSinusoidal']
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')

        self.x_dim = 2
        self.y_dim = 2

        self.q2_dB = float(self.config['q2_dB'])
        self.q2 = torch.tensor(10**(self.q2_dB/10), device=self.device)
        self.v_dB = float(self.config['v_dB'])
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim, device=self.device)
        self.cov_r = self.r2 * torch.eye(self.y_dim, device=self.device)

        self.alpha = 0.9
        self.beta = 1.1
        self.phi = 0.1 * math.pi
        self.delta = 0.01
        self.a = 1
        self.b = 1
        self.c = 0
        
        self.init_state = torch.tensor([1., 0.]).reshape((-1, 1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim), device=self.device)

    def generate_data(self):
        self.save_path = './.data/syntheticSinusoidal/'
        if self.mode == 'train':
            self.seq_len = int(self.config['train_seq_len'])
            self.num_data = int(self.config['train_seq_num'])
            self.save_path += 'train/'
        elif self.mode == 'valid':
            self.seq_len = int(self.config['valid_seq_len'])
            self.num_data = int(self.config['valid_seq_num'])
            self.save_path += 'valid/'
        elif self.mode == 'test':
            self.seq_len = int(self.config['test_seq_len'])
            self.num_data = int(self.config['test_seq_num'])
            self.save_path += 'test/'
        else:
            raise NotImplementedError()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))

        with torch.no_grad():
            for i in range(self.num_data):
                theta = torch.distributions.uniform.Uniform(0, 2*torch.pi).sample()
                dr = torch.tensor(1.)
                init_state = torch.sqrt(dr) * torch.tensor([torch.cos(theta), torch.sin(theta)]).reshape((-1, 1))
                if i % 100 == 0:
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                
                state_last = torch.clone(init_state)

                for j in range(self.seq_len):
                    x = self.next_state(state_last)
                    state_last = torch.clone(x)
                    y = self.observe(x)
                    state_mtx[i, :, j] = x.reshape(-1)
                    obs_mtx[i, :, j] = y.reshape(-1)

        torch.save(state_mtx.detach().cpu(), self.save_path + 'state.pt')
        torch.save(obs_mtx.detach().cpu(), self.save_path + 'obs.pt')
    
    def f(self, x):
        return self.alpha * torch.sin(self.beta * x + self.phi) + self.delta

    def g(self, x):
        return self.a * (self.b * x + self.c) ** 2

    def Jacobian_f(self, x):
        x = x.reshape(-1)
        df_dx0 = self.alpha * self.beta * torch.cos(self.beta * x[0] + self.phi)
        df_dx1 = self.alpha * self.beta * torch.cos(self.beta * x[1] + self.phi)
        
        jacobian = torch.zeros((2, 2), device=x.device)
        jacobian[0, 0] = df_dx0
        jacobian[1, 1] = df_dx1
        return jacobian

    def Jacobian_g(self, x):
        x = x.reshape(-1)
        u = self.b * x + self.c
        df_dx0 = 2 * self.a * self.b * u[0]
        df_dx1 = 2 * self.a * self.b * u[1]
        
        jacobian = torch.zeros((2, 2), device=x.device)
        jacobian[0, 0] = df_dx0
        jacobian[1, 1] = df_dx1
        return jacobian


# ==============================================================================
# 2. 正弦不匹配模型 (SyntheticSinusoidalMismatchModel)
# ==============================================================================
class SyntheticSinusoidalMismatchModel(GSSModel):
    def __init__(self, mode='train', EKF=False, device=None):
        super().__init__(device=device)
        self.config = config['SyntheticSinusoidal']
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')
            
        self.x_dim = 2
        self.y_dim = 2

        self.q2_dB = float(self.config['q2_dB'])
        self.q2 = torch.tensor(10**(self.q2_dB/10), device=self.device)
        self.v_dB = float(self.config['v_dB'])
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim, device=self.device)
        self.cov_r = self.r2 * torch.eye(self.y_dim, device=self.device)

        self.alpha = 1.1
        self.beta = 0.9
        self.phi = -0.1 * math.pi
        self.delta = -0.01
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.init_state = torch.tensor([1., 0.]).reshape((-1, 1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim), device=self.device)

    def generate_data(self):
        self.alpha = 0.9
        self.beta = 1.1
        self.phi = 0.1 * math.pi
        self.delta = 0.01
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.save_path = './.data/syntheticSinusoidal/'
        if self.mode == 'train':
            self.seq_len = int(self.config['train_seq_len'])
            self.num_data = int(self.config['train_seq_num'])
            self.save_path += 'train/'
        elif self.mode == 'valid':
            self.seq_len = int(self.config['valid_seq_len'])
            self.num_data = int(self.config['valid_seq_num'])
            self.save_path += 'valid/'
        elif self.mode == 'test':
            self.seq_len = int(self.config['test_seq_len'])
            self.num_data = int(self.config['test_seq_num'])
            self.save_path += 'test/'
        else:
            raise NotImplementedError()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))

        with torch.no_grad():
            for i in range(self.num_data):
                theta = torch.distributions.uniform.Uniform(0, 2*torch.pi).sample()
                dr = torch.tensor(1.)
                init_state = torch.sqrt(dr) * torch.tensor([torch.cos(theta), torch.sin(theta)]).reshape((-1, 1))
                if i % 100 == 0:
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                
                state_last = torch.clone(init_state)

                for j in range(self.seq_len):
                    x = self.next_state(state_last)
                    state_last = torch.clone(x)
                    y = self.observe(x)
                    state_mtx[i, :, j] = x.reshape(-1)
                    obs_mtx[i, :, j] = y.reshape(-1)

        torch.save(state_mtx.detach().cpu(), self.save_path + 'state.pt')
        torch.save(obs_mtx.detach().cpu(), self.save_path + 'obs.pt')
    
    def f(self, x):
        return self.alpha * torch.sin(self.beta * x + self.phi) + self.delta

    def g(self, x):
        return self.a * (self.b * x + self.c) ** 2

    def Jacobian_f(self, x):
        df_dx0 = self.alpha * self.beta * torch.cos(self.beta * x[0] + self.phi)
        df_dx1 = self.alpha * self.beta * torch.cos(self.beta * x[1] + self.phi)
        
        jacobian = torch.zeros((2, 2), device=x.device, dtype=x.dtype)
        jacobian[0, 0] = df_dx0
        jacobian[1, 1] = df_dx1
        return jacobian

    def Jacobian_g(self, x):
        x = x.reshape(-1)
        u = self.b * x + self.c
        df_dx0 = 2 * self.a * self.b * u[0]
        df_dx1 = 2 * self.a * self.b * u[1]
        
        jacobian = torch.zeros((2, 2), device=x.device, dtype=x.dtype)
        jacobian[0, 0] = df_dx0
        jacobian[1, 1] = df_dx1
        return jacobian


# ==============================================================================
# 3. Lorenz 匹配模型 (LorenzModel)
# ==============================================================================
class LorenzModel(GSSModel):
    def __init__(self, mode='train', device=None):
        super().__init__(device=device)
        self.config = config['Lorenz']
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')
        self.alpha = float(self.config['alpha'])
        self.delta = float(self.config['delta'])
        self.Taylor = int(self.config['Taylor'])
        self.is_linear = bool(int(self.config['is_linear']))
        self.x_dim = 3
        self.y_dim = 3
        self.q2_dB = float(self.config['q2_dB'])
        self.q2 = torch.tensor(10**(self.q2_dB/10), device=self.device)
        self.v_dB = float(self.config['v_dB'])
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim, device=self.device)
        self.cov_r = self.r2 * torch.eye(self.y_dim, device=self.device)

        self.theta = 10 * 2 * math.pi/360

        self.init_state = torch.tensor([1., 0., 0.], device=self.device).reshape((-1, 1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim), device=self.device)

    def generate_data(self):
        self.save_path = './.data/syntheticNL/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      

        if self.mode == 'train':
            self.seq_len = int(self.config['train_seq_len'])
            self.num_data = int(self.config['train_seq_num'])
            self.save_path += 'train/'
        elif self.mode == 'valid':
            self.seq_len = int(self.config['valid_seq_len'])
            self.num_data = int(self.config['valid_seq_num'])
            self.save_path += 'valid/'
        elif self.mode == 'test':
            self.seq_len = int(self.config['test_seq_len'])
            self.num_data = int(self.config['test_seq_num'])
            self.save_path += 'test/'
        else:
            raise NotImplementedError()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))
        with torch.no_grad():
            for i in range(self.num_data):
                self.init_state = torch.zeros((self.x_dim, 1))
                if i % 100 == 0:
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                state_tmp = torch.zeros((self.x_dim, self.seq_len))
                obs_tmp = torch.zeros((self.y_dim, self.seq_len))
                state_last = torch.clone(self.init_state)

                for j in range(self.seq_len):
                    x = self.next_state(state_last)
                    state_last = torch.clone(x)
                    y = self.observe(x)
                    state_tmp[:, j] = x.reshape(-1)
                    obs_tmp[:, j] = y.reshape(-1)
                
                state_mtx[i] = state_tmp
                obs_mtx[i] = obs_tmp
        
        torch.save(state_mtx.detach().cpu(), self.save_path + 'state.pt')
        torch.save(obs_mtx.detach().cpu(), self.save_path + 'obs.pt')                 

    def f(self, x):
        F_mat = self.Jacobian_f(x).to(device=x.device, dtype=x.dtype)
        return torch.matmul(F_mat, x)

    def g(self, x):
        if self.is_linear:
            return x.reshape((-1, 1))
        else:
            x = x.reshape(-1)
            r = torch.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            theta = torch.arccos(x[2] / (r + 1e-9))
            phi = torch.arctan2(x[1], x[0])
            return torch.stack([r, theta, phi]).reshape(-1, 1)

    def Jacobian_f(self, x):
        A = np.array([
            [-(10 + 25*self.alpha), (10 + 25*self.alpha), 0],
            [(28 -  35*self.alpha), (29*self.alpha - 1), -x[0].item()],
            [0, x[0].item(), -(8.0 + self.alpha)/3]
        ])
        F = np.eye(self.x_dim)
        for j in range(1, self.Taylor+1):
            F += np.linalg.matrix_power(A * self.delta, j) / math.factorial(j)
        return torch.tensor(F, dtype=x.dtype, device=x.device)
        
    def Jacobian_g(self, x):
        if self.is_linear:
            return torch.eye(self.y_dim, device=x.device, dtype=x.dtype)
        else:
            x = x.reshape(-1)
            eps = 1e-9
            r = torch.sqrt(x[0]**2 + x[1]**2 + x[2]**2).clamp_min(eps)
            J11 = x[0] / r
            J12 = x[1] / r
            J13 = x[2] / r
            J21 = -x[0] * x[2] / (r**2 * torch.sqrt(1 - (x[2]/r)**2 + eps))
            J22 = -x[1] * x[2] / (r**2 * torch.sqrt(1 - (x[2]/r)**2 + eps))
            J23 = torch.sqrt(x[0]**2 + x[1]**2) / (r**2 + eps)
            denom3 = (x[0]**2 + x[1]**2).clamp_min(eps)
            J31 = -x[1] / denom3
            J32 =  x[0] / denom3
            J33 = x.new_tensor(0.)
            return torch.stack([
                torch.stack([J11, J12, J13]),
                torch.stack([J21, J22, J23]),
                torch.stack([J31, J32, J33])
            ])  


# ==============================================================================
# 4. Lorenz 不匹配模型 (LorenzMismatchModel)
# ==============================================================================
class LorenzMismatchModel(GSSModel):
    def __init__(self, mode='train', device=None):
        super().__init__(device=device)
        self.config = config['Lorenz']
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')
        self.alpha = float(self.config['alpha'])
        self.delta = float(self.config['delta'])
        self.Taylor = int(self.config['Taylor'])
        self.is_linear = bool(int(self.config['is_linear']))
        self.x_dim = 3
        self.y_dim = 3
        self.beta = float(self.config['beta'])
        self.q2_dB = float(self.config['q2_dB'])
        self.q2 = torch.tensor(10**(self.q2_dB/10), device=self.device)
        self.v_dB = float(self.config['v_dB'])
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim, device=self.device)
        self.cov_r = self.r2 * torch.eye(self.y_dim, device=self.device)
        self.mismatch_f = 1
        self.theta = 0
        self.init_state = torch.tensor([1., 0., 0.], device=self.device).reshape((-1, 1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim), device=self.device)
        self.H_true = torch.eye(self.y_dim, device=self.device, dtype=torch.float32)
        self.dist_q = torch.distributions.MultivariateNormal(
            torch.zeros(self.x_dim, device=self.device), 
            self.cov_q.to(self.device)
        )
        self.dist_r = torch.distributions.MultivariateNormal(
            torch.zeros(self.y_dim, device=self.cov_r.device), 
            self.cov_r
        )

    def generate_data(self):
        theta_rad = math.radians(float(self.config['theta']))
        self.axis = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        u = self.axis.to(torch.float32)
        u = u / torch.norm(u)
        ux, uy, uz = u
        c = math.cos(theta_rad)
        s = math.sin(theta_rad) 
        one_c = 1.0 - c
        self.H_true = torch.tensor([
            [c + ux*ux*one_c,      ux*uy*one_c - uz*s,  ux*uz*one_c + uy*s],
            [uy*ux*one_c + uz*s,   c + uy*uy*one_c,     uy*uz*one_c - ux*s],
            [uz*ux*one_c - uy*s,   uz*uy*one_c + ux*s,  c + uz*uz*one_c    ],
        ], dtype=torch.float32, device=self.device)
        
        self.save_path = './.data/syntheticNL/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      
        if self.mode == 'train':
            self.seq_len = int(self.config['train_seq_len'])
            self.num_data = int(self.config['train_seq_num'])
            self.save_path += 'train/'
        elif self.mode == 'valid':
            self.seq_len = int(self.config['valid_seq_len'])
            self.num_data = int(self.config['valid_seq_num'])
            self.save_path += 'valid/'
        elif self.mode == 'test':
            self.seq_len = int(self.config['test_seq_len'])
            self.num_data = int(self.config['test_seq_num'])
            self.save_path += 'test/'
        else:
            raise NotImplementedError()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len), device=self.device)
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len), device=self.device)
        mismatch_ratio = 0

        sigma_true = 10.0 * (1 + mismatch_ratio)
        rho_true = 28.0 * (1 + mismatch_ratio)
        beta_true = (8.0 / 3.0) * (1 + mismatch_ratio)
        with torch.no_grad():
            for i in range(self.num_data):
                self.init_state = torch.zeros((self.x_dim, 1), device=self.device)
                if i % 100 == 0:
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                state_tmp = torch.zeros((self.x_dim, self.seq_len), device=self.device)
                obs_tmp = torch.zeros((self.y_dim, self.seq_len), device=self.device)
                state_last = torch.clone(self.init_state)

                for j in range(self.seq_len):
                    x_0 = state_last[0, 0]
                    A_true = torch.tensor([
                        [-sigma_true, sigma_true, 0.0],
                        [rho_true, -1.0, -x_0],
                        [0.0, x_0, -beta_true]
                    ], device=self.device, dtype=torch.float32)
                    
                    F_true = torch.eye(self.x_dim, device=self.device, dtype=torch.float32)
                    term = torch.eye(self.x_dim, device=self.device, dtype=torch.float32)
                    
                    for k in range(1, 5 + 1):
                        term = term @ (A_true * 0.02) / k
                        F_true = F_true + term
                
                    noise_tensor = self.dist_q.sample().reshape(-1, 1)
                    x = torch.matmul(F_true, state_last) + noise_tensor
                    state_last = torch.clone(x)

                    noise_tensor = self.dist_r.sample().reshape(-1, 1)
                    y = torch.matmul(self.H_true, x).reshape(-1, 1) + noise_tensor
                    state_tmp[:, j] = x.reshape(-1)
                    obs_tmp[:, j] = y.reshape(-1)
                
                state_mtx[i] = state_tmp
                obs_mtx[i] = obs_tmp
        
        torch.save(state_mtx.detach().cpu(), self.save_path + 'state.pt')
        torch.save(obs_mtx.detach().cpu(), self.save_path + 'obs.pt')                 

    def f(self, x):
        F_mat = self.Jacobian_f(x).to(device=x.device, dtype=x.dtype)
        return torch.matmul(F_mat, x)                         
    
    def g(self, x):
        if self.is_linear:
            return x.reshape((-1, 1))
        else:
            x = x.reshape(-1)
            r = torch.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            theta = torch.arccos(x[2] / (r + 1e-9))
            phi = torch.arctan2(x[1], x[0])
            return torch.stack([r, theta, phi]).reshape(-1, 1)

    def Jacobian_f(self, x):
        A = np.array([
            [-(10 + 25*self.alpha), (10 + 25*self.alpha), 0],
            [(28 -  35*self.alpha), (29 * self.alpha - 1), -x[0].item()],
            [0, x[0].item(), -(8.0 + self.alpha)/3]
        ]) * (1 + self.mismatch_f * self.beta)
        F = np.eye(self.x_dim)
        for j in range(1, self.Taylor+1):
            F += np.linalg.matrix_power(A * self.delta, j) / math.factorial(j)
        return torch.tensor(F, dtype=x.dtype, device=x.device)
        
    def Jacobian_g(self, x):
        if self.is_linear:
            return torch.eye(self.y_dim, device=x.device, dtype=x.dtype)
        else:
            x = x.reshape(-1)
            eps = 1e-9
            r = torch.sqrt(x[0]**2 + x[1]**2 + x[2]**2).clamp_min(eps)
            J11 = x[0] / r
            J12 = x[1] / r
            J13 = x[2] / r
            J21 = -x[0] * x[2] / (r**2 * torch.sqrt(1 - (x[2]/r)**2 + eps))
            J22 = -x[1] * x[2] / (r**2 * torch.sqrt(1 - (x[2]/r)**2 + eps))
            J23 = torch.sqrt(x[0]**2 + x[1]**2) / (r**2 + eps)
            denom3 = (x[0]**2 + x[1]**2).clamp_min(eps)
            J31 = -x[1] / denom3
            J32 = x[0] / denom3
            J33 = x.new_tensor(0.)
            return torch.stack([
                torch.stack([J11, J12, J13]),
                torch.stack([J21, J22, J23]),
                torch.stack([J31, J32, J33])
            ])