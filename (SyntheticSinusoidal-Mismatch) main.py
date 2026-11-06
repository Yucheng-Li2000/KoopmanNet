import os
import configparser
import numpy as np
import torch

from GSSFiltering.model import SyntheticSinusoidalMismatchModel
from GSSFiltering.filtering import DataDriven_Koopman_Kalman_Filter
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not os.path.exists('./.results'):
    os.mkdir('./.results')

config = configparser.ConfigParser()
config.read('./config.ini')

TRAIN = True
# TRAIN = False

n_features = int(config['SyntheticSinusoidal'].get('n_features', '12'))
train_iter = int(config['Train']['train_iter'])
test_list = [str(train_iter)]

# ---- 损失与推理性能记录 ----
loss_list_Koopman = []
valid_loss_Koopman = []

time_list_Koopman_ms = []                     # 推理时间（每步，ms）
train_time_Koopman_forward_ms = []            # 训练前向时间（每步，ms）
train_time_Koopman_forward_mean_per_repeat = []

num_repeat = 5

for repeat_idx in range(num_repeat):
    print(f"\n=== 第 {repeat_idx+1} 次训练与测试 ===")

    train_time_Koopman_forward_this_repeat = []

    # 1. 生成不匹配数据
    if TRAIN:
        SyntheticSinusoidalMismatchModel(mode='train').generate_data()
        SyntheticSinusoidalMismatchModel(mode='valid').generate_data()
    SyntheticSinusoidalMismatchModel(mode='test').generate_data()

    # 2. 训练阶段
    if TRAIN:
        trainer_koopman = Trainer(
            dnn=DataDriven_Koopman_Kalman_Filter(
                SyntheticSinusoidalMismatchModel(mode='train'), n_features
            ),
            data_path='./.data/syntheticSinusoidal/train/',
            save_path='(syntheticSinusoidalMM) KoopmanKalmanNet.pt'
        )

        koopman_filter_valid = DataDriven_Koopman_Kalman_Filter(
            SyntheticSinusoidalMismatchModel(mode='valid'), n_features
        )

        for i in range(train_iter):
            t_koop_fwd = trainer_koopman.train_batch()
            if t_koop_fwd is None:
                t_koop_fwd = getattr(trainer_koopman, 'last_train_forward_ms', float('nan'))
            train_time_Koopman_forward_this_repeat.append(float(t_koop_fwd))
            train_time_Koopman_forward_ms.append(float(t_koop_fwd))

            trainer_koopman.dnn.reset(clean_history=True)

            # 定期在验证集上评估并保存最佳权重
            if trainer_koopman.train_count % trainer_koopman.save_num == 0:
                trainer_koopman.validate(
                    Tester(
                        filter=koopman_filter_valid,
                        data_path='./.data/syntheticSinusoidal/valid/',
                        model_path='./.model_saved/(syntheticSinusoidalMM) KoopmanKalmanNet_'
                                   + str(trainer_koopman.train_count) + '.pt',
                        is_validation=True
                    )
                )
                valid_loss_Koopman.append(trainer_koopman.valid_loss)

        if len(train_time_Koopman_forward_this_repeat) > 0:
            train_time_Koopman_forward_mean_per_repeat.append(
                float(np.mean(train_time_Koopman_forward_this_repeat))
            )

        np.save('./.results/valid_loss_koopman_mismatch.npy', np.array(valid_loss_Koopman))

    # 3. 测试阶段
    print('\n=== 测试阶段 ===')
    koopman_filter_test = DataDriven_Koopman_Kalman_Filter(
        SyntheticSinusoidalMismatchModel(mode='test'), n_features
    )

    for elem in test_list:
        tester_koopman = Tester(
            filter=koopman_filter_test,
            data_path='./.data/syntheticSinusoidal/test/',
            model_path='./.model_saved/(syntheticSinusoidalMM) KoopmanKalmanNet_' + elem + '.pt'
        )
        loss_list_Koopman.append(tester_koopman.loss.item())
        time_list_Koopman_ms.append(tester_koopman.step_time_mean_ms)


# ========== 汇总统计与打印 ==========
def stat_str(arr):
    if len(arr) == 0:
        return "nan/nan/nan"
    m = round(np.mean(arr), 4)
    s = round(np.std(arr, ddof=1), 4) if len(arr) > 1 else 0.0
    p95 = round(np.percentile(arr, 95), 4)
    return f"{m}/{s}/{p95}"


print("\n(SyntheticSinusoidal Mismatch):")
print(
    "Koopman losses [dB]:",
    [round(x, 2) for x in loss_list_Koopman],
    " mean:",
    round(float(np.mean(loss_list_Koopman)) if len(loss_list_Koopman) > 0 else float('nan'), 4)
)

print("\nPer-step inference time [ms] (mean/std/p95):")
print("Koopman:", stat_str(time_list_Koopman_ms))

print("\nTraining forward time [ms] (mean/std/p95):")
print("Koopman train-forward:", stat_str(train_time_Koopman_forward_ms))