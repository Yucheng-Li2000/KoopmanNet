import os
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GSSFiltering.model import LorenzModel
from GSSFiltering.filtering import DataDriven_Koopman_Kalman_Filter
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for folder in ('./.results', './.model_saved', './.data/syntheticNL/train', './.data/syntheticNL/valid', './.data/syntheticNL/test'):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# ================== 设备配置 (CUDA) ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前运行设备: {device}")
if device.type == 'cuda':
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

def to_device(filter_obj, dev):
    if hasattr(filter_obj, 'to') and callable(filter_obj.to):
        return filter_obj.to(dev)
    return filter_obj

config = configparser.ConfigParser()
config.read('./config.ini')

TRAIN = True

train_iter = int(config['Train']['train_iter'])
test_list = [str(train_iter)]

# 结果记录
loss_koopman_match = []
valid_loss_koopman = []
time_koopman_ms = []

num_repeat = 1
n_features = int(config['Lorenz'].get('n_features', '12')) if 'Lorenz' in config else 12

koopman_filter_test = to_device(DataDriven_Koopman_Kalman_Filter(LorenzModel(mode='test'), n_features), device)

for repeat_idx in range(num_repeat):
    print(f"\n=== 第 {repeat_idx+1} 次训练与测试 ===")

    # 1. 生成数据
    if TRAIN:
        LorenzModel(mode='train').generate_data()
        LorenzModel(mode='valid').generate_data()
    LorenzModel(mode='test').generate_data()

    # 2. 训练阶段
    if TRAIN:
        dnn_koopman = to_device(DataDriven_Koopman_Kalman_Filter(LorenzModel(mode='train'), n_features), device)
        trainer_koopman = Trainer(
            dnn=dnn_koopman,
            data_path='./.data/syntheticNL/train/',
            save_path='(lorenz) Koopman.pt'
        )

        for i in range(train_iter):
            t_koop_fwd = trainer_koopman.train_batch()
            if t_koop_fwd is None:
                t_koop_fwd = getattr(trainer_koopman, 'last_train_forward_ms', float('nan'))
            trainer_koopman.dnn.reset(clean_history=True)

            # 定期在验证集上评估并保存最佳权重
            if trainer_koopman.train_count % trainer_koopman.save_num == 0:
                valid_koop = to_device(DataDriven_Koopman_Kalman_Filter(LorenzModel(mode='valid'), n_features), device)
                trainer_koopman.validate(
                    Tester(
                        filter=valid_koop,
                        data_path='./.data/syntheticNL/valid/',
                        model_path=f'./.model_saved/(lorenz) Koopman_{trainer_koopman.train_count}.pt',
                        is_validation=True
                    )
                )
                valid_loss_koopman.append(trainer_koopman.valid_loss)

        np.save('./.results/valid_loss_koopman_lorenz.npy', np.array(valid_loss_koopman))

    # 3. 测试阶段
    print('\n=== 测试阶段 ===')
    for step in test_list:
        tester_koop = Tester(
            filter=koopman_filter_test,
            data_path='./.data/syntheticNL/test/',
            model_path=f'./.model_saved/(lorenz) Koopman_{step}.pt'
        )
        loss_koopman_match.append(tester_koop.loss.item())
        time_koopman_ms.append(tester_koop.step_time_mean_ms)

# ========== 结果输出 ==========
print("Koopman (match):", [round(x, 2) for x in loss_koopman_match])
np.save('./.results/lorenz_loss_koopman_match.npy', np.array(loss_koopman_match))

mean_koop = float(np.mean(np.array(loss_koopman_match))) if len(loss_koopman_match) > 0 else float('nan')
print("Koopman mean [dB]:", round(mean_koop, 2))

if len(time_koopman_ms) > 0:
    print(f"Koopman inference time mean: {np.mean(time_koopman_ms):.4f} ms")


# ================== 绘制测试集 Lorenz 轨迹 (True vs Koopman) ==================

def to_T3_from_tensor(x_seq: torch.Tensor) -> np.ndarray:
    """提取第 0 条轨迹的前三维物理状态 [T, 3]"""
    traj = x_seq[0]
    traj = traj.permute(1, 0)
    traj = traj.detach().cpu().numpy()
    return traj[:, :3]


test_data_path = './.data/syntheticNL/test/'

# 加载真实轨迹数据与估计轨迹
true_x = torch.load(os.path.join(test_data_path, 'state.pt'), map_location='cpu')
true_traj = to_T3_from_tensor(true_x)

path_koopman = os.path.join(test_data_path, 'DD-KKF x_hat.pt')
if not os.path.exists(path_koopman):
    path_koopman = os.path.join(test_data_path, 'Koopman x_hat.pt')

xhat_koopman = torch.load(path_koopman, map_location='cpu') if os.path.exists(path_koopman) else None

data_list = [true_traj]
titles = ["True"]

if xhat_koopman is not None:
    data_list.append(to_T3_from_tensor(xhat_koopman))
    titles.append("KoopmanNet")

fig, axs = plt.subplots(1, max(2, len(data_list)), figsize=(min(21, 3.5 * len(data_list)), 3), subplot_kw={'projection': '3d'})
if len(data_list) == 1:
    axs = np.array([axs])

x_min, x_max = true_traj[:, 0].min(), true_traj[:, 0].max()
y_min, y_max = true_traj[:, 1].min(), true_traj[:, 1].max()
z_min, z_max = true_traj[:, 2].min(), true_traj[:, 2].max()

color_map = {
    "True": '#222222',       # 黑色
    "KoopmanNet": '#d62728', # 红色
}
colors = [color_map.get(t, '#333333') for t in titles]

idx = 0
for ax, x_array, title, color in zip(axs.flat, data_list, titles, colors):
    ax.set_visible(True)
    ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2], color=color, linewidth=1.2)
    ax.set_xlabel('$x_1$', fontsize=8, fontname='Times New Roman')
    ax.set_ylabel('$x_2$', fontsize=8, fontname='Times New Roman')
    ax.set_zlabel('$x_3$', fontsize=8, fontname='Times New Roman')
    subplot_label = f"({chr(97+idx)}) {title}"
    ax.set_title(subplot_label, fontsize=16, pad=2, fontname='Times New Roman', y=0.20, loc='center')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    idx += 1

# 隐藏可能存在的多余空子图
for ax in axs.flat[idx:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("./.results/lorenz_koopman_match_trajectories.eps", format="eps", dpi=300, bbox_inches='tight', pad_inches=0.00)
plt.savefig("./.results/lorenz_koopman_match_trajectories.png", format="png", dpi=300, bbox_inches='tight', pad_inches=0.00)

# 分别保存每张单图
for i, (x_array, title, color) in enumerate(zip(data_list, titles, colors)):
    fig_single = plt.figure(figsize=(4, 3))
    ax_single = fig_single.add_subplot(111, projection='3d')

    ax_single.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2], color=color, linewidth=1.2)
    ax_single.set_xlabel('$x_1$', fontsize=8, fontname='Times New Roman')
    ax_single.set_ylabel('$x_2$', fontsize=8, fontname='Times New Roman')
    ax_single.set_zlabel('$x_3$', fontsize=8, fontname='Times New Roman')

    subplot_label = f"({chr(97+i)}) {title}"
    ax_single.set_title(subplot_label, fontsize=16, pad=2, fontname='Times New Roman', y=0.20, loc='center')

    ax_single.set_xlim(x_min, x_max)
    ax_single.set_ylim(y_min, y_max)
    ax_single.set_zlim(z_min, z_max)
    ax_single.set_xticks([])
    ax_single.set_yticks([])
    ax_single.set_zticks([])
    ax_single.grid(False)
    ax_single.set_axis_off()

    fig_single.tight_layout()
    fig_single.savefig(f"./.results/lorenz_koopman_match_{i+1}_{title}.eps", format="eps", dpi=300, bbox_inches='tight', pad_inches=0.00)
    fig_single.savefig(f"./.results/lorenz_koopman_match_{i+1}_{title}.png", format="png", dpi=300, bbox_inches='tight', pad_inches=0.00)
    plt.close(fig_single)

plt.show()