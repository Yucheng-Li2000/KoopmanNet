import os
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GSSFiltering.model import LorenzMismatchModel
from GSSFiltering.filtering import DataDriven_Koopman_Kalman_Filter
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for folder in ('./.results', './.model_saved', './.data/syntheticNL/train', './.data/syntheticNL/valid', './.data/syntheticNL/test'):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

config = configparser.ConfigParser()
config.read('./config.ini')

TRAIN = True
n_runs = 10
n_features = 12

all_test_loss_koopman = []

for run in range(n_runs):
    print(f"\n=== Run {run+1}/{n_runs} ===")
    
    # 1. 生成数据
    if TRAIN:
        LorenzMismatchModel(mode='train').generate_data()
        LorenzMismatchModel(mode='valid').generate_data()
    LorenzMismatchModel(mode='test').generate_data()

    train_iter = int(config['Train']['train_iter'])
    test_list = [str(train_iter)]

    valid_loss_koopman = []
    test_loss_koopman = []

    # 2. 训练阶段
    if TRAIN:
        trainer_koopman = Trainer(
            dnn=DataDriven_Koopman_Kalman_Filter(LorenzMismatchModel(mode='train'), n_features),
            data_path='./.data/syntheticNL/train/',
            save_path='(LorenzMismatchModel) Koopman.pt'
        )

        koopman_filter_valid = DataDriven_Koopman_Kalman_Filter(LorenzMismatchModel(mode='valid'), n_features)

        for _ in range(train_iter):
            trainer_koopman.train_batch()
            trainer_koopman.dnn.reset(clean_history=True)

            # 定期在验证集上评估并保存最佳模型
            if trainer_koopman.train_count % trainer_koopman.save_num == 0:
                trainer_koopman.validate(
                    Tester(
                        filter=koopman_filter_valid,
                        data_path='./.data/syntheticNL/valid/',
                        model_path=f'./.model_saved/(LorenzMismatchModel) Koopman_{trainer_koopman.train_count}.pt',
                        is_validation=True,
                    )
                )
                valid_loss_koopman.append(trainer_koopman.valid_loss)

        # 保存验证损失曲线
        np.save(f'./.results/valid_loss_koopman_run{run}.npy', np.array(valid_loss_koopman))

    # 3. 测试阶段
    print('\n=== 测试阶段 ===')
    koopman_filter_test = DataDriven_Koopman_Kalman_Filter(LorenzMismatchModel(mode='test'), n_features)

    for step in test_list:
        koopman_tester = Tester(
            filter=koopman_filter_test,
            data_path='./.data/syntheticNL/test/',
            model_path=f'./.model_saved/(LorenzMismatchModel) Koopman_{step}.pt',
        )
        test_loss_koopman.append(koopman_tester.loss.item())

    all_test_loss_koopman.append(test_loss_koopman[0] if len(test_loss_koopman) > 0 else float('nan'))


# ========== 汇总统计与输出 ==========
def print_loss_list_and_mean(name, loss_list):
    formatted_list = [f"{x:.2f}" for x in loss_list]
    print(f"{name} test loss list: {formatted_list}")
    print(f"{name} test loss mean: {np.mean(loss_list):.2f}")


print("\n=== 多次实验最终测试结果 ===")
print_loss_list_and_mean("Koopman", all_test_loss_koopman)


# ================== 绘制测试集 3D Lorenz 轨迹 (True vs Koopman) ==================

def to_T3_from_tensor(x_seq: torch.Tensor) -> np.ndarray:
    """提取第 0 条轨迹的前三维物理状态 [T, 3]"""
    traj = x_seq[0].permute(1, 0).cpu().numpy()
    return traj[:, :3]


test_data_path = './.data/syntheticNL/test/'

# 加载真实轨迹与 Koopman 估计轨迹
true_x = torch.load(os.path.join(test_data_path, 'state.pt'))
true_traj = to_T3_from_tensor(true_x)

path_koopman = os.path.join(test_data_path, 'DD-KKF x_hat.pt')
xhat_koopman = torch.load(path_koopman) if os.path.exists(path_koopman) else None

data_list = [true_traj]
titles = ["True"]

if xhat_koopman is not None:
    data_list.append(to_T3_from_tensor(xhat_koopman))
    titles.append("Koopman")

num_subplots = len(data_list)
fig, axs = plt.subplots(1, max(2, num_subplots), figsize=(3.5 * num_subplots, 3), subplot_kw={'projection': '3d'})
if num_subplots == 1:
    axs = np.array([axs])

x_min, x_max = true_traj[:, 0].min(), true_traj[:, 0].max()
y_min, y_max = true_traj[:, 1].min(), true_traj[:, 1].max()
z_min, z_max = true_traj[:, 2].min(), true_traj[:, 2].max()

color_dict = {
    'True': '#222222',       # 黑色
    'Koopman': '#8b3fc9',    # 紫色
}
colors = [color_dict.get(t, '#000000') for t in titles]

idx = 0
for i, (ax, x_array, title) in enumerate(zip(axs.flat, data_list, titles)):
    color = colors[i]
    ax.set_visible(True)
    ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2], color=color, linewidth=1.2)
    ax.set_xlabel('$x_1$', fontsize=8, fontname='Times New Roman')
    ax.set_ylabel('$x_2$', fontsize=8, fontname='Times New Roman')
    ax.set_zlabel('$x_3$', fontsize=8, fontname='Times New Roman')
    subplot_label = f"({chr(97+i)}) {title}"
    ax.set_title(subplot_label, fontsize=14, pad=2, fontname='Times New Roman', y=0.20, loc='center')
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
plt.savefig("./.results/lorenz_koopman_trajectories.eps", format="eps", dpi=900, bbox_inches='tight', pad_inches=0.00)
plt.savefig("./.results/lorenz_koopman_trajectories.png", format="png", dpi=900, bbox_inches='tight', pad_inches=0.00)
plt.show()