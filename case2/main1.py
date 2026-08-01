import os
import torch
import argparse
import numpy as np
import pickle
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import sys
import torch.serialization
import random


from OPFT import JiT_MODELS
from utils import EMA
from train_functions import train_main
from generate import generate_samples


def set_gen_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    image_size = (60, 60)
    channels = 1
    cond_dim = 500
    ts_feature = [50, 10]
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1e-8
    train_num = [0, 1000]
    test_num = [1000, 1200]
    beta = "linear"
    gen_seed = 42


def parse_patch_size(s):

    try:
        return tuple(map(int, s.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "patch_size必须是用逗号分隔的两个整数，如'16,16'"
        )


def get_params():
    parser = argparse.ArgumentParser(
        description="JiT for flow matching oil reservoir inversion (No PCA)"
    )

    parser.add_argument("--batchsize", type=int, default=32, help="批大小")
    parser.add_argument("--numworkers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--inch", type=int, default=1, help="输入通道数")
    parser.add_argument("--cdim", type=int, default=500, help="条件观测数据维度")
    parser.add_argument(
        "--model_name", type=str, default="jit_small", help="JiT模型类型"
    )
    parser.add_argument(
        "--patch_size", type=parse_patch_size, default="20,20", help="图像分块大小"
    )
    parser.add_argument("--attn_drop", type=float, default=0, help="注意力层dropout率")
    parser.add_argument(
        "--proj_drop", type=float, default=0, help="注意力投影层dropout率"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--epochs", type=int, default=500, help="训练总轮数")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA权重衰减率")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW权重衰减"
    )
    parser.add_argument(
        "--t_eps", type=float, default=5e-2, help="流匹配损失中的数值稳定项"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./model_ckpt", help="模型权重保存目录"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--rank", type=int, default=0, help="分布式训练排名")
    parser.add_argument("--local_rank", default=-1, type=int, help="分布式训练节点排名")

    parser.add_argument("--P_mean", type=float, default=-0.8, help="时间步采样均值参数")
    parser.add_argument("--P_std", type=float, default=0.8, help="时间步采样标准差参数")
    parser.add_argument(
        "--label_drop_prob", type=float, default=0.15, help="标签丢弃概率"
    )
    parser.add_argument("--cfg_scale", type=float, default=1, help="CFG引导强度")
    parser.add_argument(
        "--cfg_interval_min", type=float, default=0.0, help="CFG生效的最小时间步"
    )
    parser.add_argument(
        "--cfg_interval_max", type=float, default=1.0, help="CFG生效的最大时间步"
    )

    parser.add_argument("--num_steps", type=int, default=15, help="ODE积分步数")
    parser.add_argument("--noise_scale", type=float, default=1, help="初始噪声强度")
    parser.add_argument(
        "--integrate_method",
        type=str,
        default="rk4",
        choices=["euler", "heun", "rk4"],
        help="ODE积分方法",
    )
    parser.add_argument(
        "--gen_model_path",
        type=str,
        default="./model_ckpt/model_best.pth",
        help="生成时加载的模型路径",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


class SyntheticConditionalDataset(Dataset):
    def __init__(self, data_norm, cond_norm, ind=[0, 100]):
        self.images = data_norm[ind[0] : ind[1]]
        self.conditions = cond_norm[ind[0] : ind[1]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.conditions[idx]


def create_dataloader(data_norm, cond_norm, args):
    dataset = SyntheticConditionalDataset(data_norm, cond_norm, ind=Config.train_num)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        pin_memory=True if Config.device.type == "cuda" else False,
        num_workers=args.numworkers,
    )


def create_dataloader2(data_norm, cond_norm, args):
    dataset = SyntheticConditionalDataset(data_norm, cond_norm, ind=Config.test_num)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        pin_memory=True if Config.device.type == "cuda" else False,
        num_workers=args.numworkers,
    )


def initial(args):
    rootpath = os.path.dirname(os.path.abspath(__file__))
    path = rootpath

    with h5py.File(path + "/train_data.h5", "r") as f:
        data_ = f["x"][:]
        cond = f["y"][:]
    print(f"原始数据形状: data_={data_.shape}, cond={cond.shape}")
    data_ = np.log(data_ + 1)

    data = data_.reshape(-1, 1, 60, 60)
    print(f"重塑后数据形状: data={data.shape}")

    with h5py.File(rootpath + "/data/ini_data.h5", "r") as f:
        dobstrue = f["dobstrue"][:]
        true_para_ = f["true_para"][:]
    true_para_ = np.log(true_para_ + 1)

    true_para = true_para_.reshape(1, 1, 60, 60)
    true_para = torch.tensor(true_para, dtype=torch.float32)
    dobstrue = torch.tensor(dobstrue.reshape(1, -1), dtype=torch.float32)
    print(f"真实参数形状: true_para={true_para.shape}, dobstrue={dobstrue.shape}")

    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data_norm = (data - data_min) / (data_max - data_min + Config.eps)
    data_norm = np.nan_to_num(data_norm, nan=0.0)
    data_norm = np.clip(data_norm, 0, 1)
    data_norm = data_norm * 2 - 1
    data_norm = torch.tensor(data_norm, dtype=torch.float32)
    print(f"原始数据最小值: {data_min.min()}")
    print(f"原始数据最大值: {data_max.max()}")

    cond_max = np.max(cond, axis=0)
    cond_min = np.min(cond, axis=0)
    cond_norm = (cond - cond_min) / (cond_max - cond_min + Config.eps)
    cond_norm = np.nan_to_num(cond_norm, nan=0.0)
    cond_norm = np.clip(cond_norm, 0, 1)
    cond_norm = cond_norm * 2 - 1
    cond_norm = torch.tensor(cond_norm, dtype=torch.float32)

    print(f"data_norm 范围: [{data_norm.min():.4f}, {data_norm.max():.4f}]")
    print(f"cond_norm 范围: [{cond_norm.min():.4f}, {cond_norm.max():.4f}]")

    vis_dir = os.path.join(rootpath, "data")
    os.makedirs(vis_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax[0].imshow(data_norm[0, 0, :, :], cmap="jet")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title("Sample 0 (Channel 0)")
    im2 = ax[1].imshow(data_norm[1, 0, :, :], cmap="jet")
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Sample 1 (Channel 0)")
    fig.savefig(
        os.path.join(vis_dir, "data_norm_139x48.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    dobstrue_ = np.clip(dobstrue.cpu().numpy(), cond_min, cond_max)
    dobstrue_ = (dobstrue_ - cond_min) / (cond_max - cond_min + Config.eps)
    dobstrue_ = np.nan_to_num(dobstrue_, nan=0.0)
    dobstrue_ = np.clip(dobstrue_, 0, 1)
    dobstrue_norm = dobstrue_ * 2 - 1

    true_para_norm = (true_para.cpu().numpy() - data_min) / (
        data_max - data_min + Config.eps
    ) * 2 - 1
    true_para_norm = np.clip(true_para_norm, -1, 1)
    true_para_norm = torch.tensor(true_para_norm, dtype=torch.float32)

    initial_res = {}
    initial_res["dobstrue"] = dobstrue
    initial_res["dobstrue_norm"] = dobstrue_norm
    initial_res["true_para"] = true_para
    initial_res["true_para_norm"] = true_para_norm
    initial_res["data_norm"] = data_norm
    initial_res["cond_norm"] = cond_norm
    initial_res["data_max"] = data_max
    initial_res["data_min"] = data_min
    initial_res["cond_max"] = cond_max
    initial_res["cond_min"] = cond_min
    initial_res["rootpath"] = rootpath
    initial_res["path"] = path
    initial_res["dataloader"] = create_dataloader(data_norm, cond_norm, args)
    initial_res["testloader"] = create_dataloader2(data_norm, cond_norm, args)

    return initial_res


def test_generate(initial_res, args, config):

    device = config.device
    rootpath = initial_res["rootpath"]
    dobstrue_norm = initial_res["dobstrue_norm"]
    true_para_norm = initial_res["true_para_norm"]
    data_max = initial_res["data_max"]
    data_min = initial_res["data_min"]
    dataloader = initial_res["dataloader"]

    jit_model = JiT_MODELS[args.model_name](
        patch_size=args.patch_size,
        obs_dim=args.cdim,
        attn_drop=args.attn_drop,
        in_chans=config.channels,
        img_size=config.image_size,
    ).to(device)

    try:
        checkpoint = torch.load(
            args.gen_model_path, map_location=device, weights_only=False
        )

        state_dict = checkpoint["model_state_dict"]
        unwanted_keys = ["rope.cos_cached", "rope.sin_cached"]

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not any(unwanted_key in k for unwanted_key in unwanted_keys):
                filtered_state_dict[k] = v
            else:
                print(f"已过滤无效键：{k}")

        missing_keys, unexpected_keys = jit_model.load_state_dict(
            filtered_state_dict, strict=False
        )

        if missing_keys:
            print(f"警告：以下键缺失：{missing_keys}")
        if unexpected_keys:
            print(f"警告：以下键未使用：{unexpected_keys}")

        print(
            f"成功加载模型: {args.gen_model_path} | 训练轮数: {checkpoint['epoch']} | 损失: {checkpoint['loss']:.6f}"
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    jit_model.eval()

    print("\n=== 测试训练集样本生成（3个样本） ===")

    batch = next(iter(dataloader))
    x_train_orig = batch[0]
    lab_train = batch[1]

    n_show = 3
    x_train_orig = x_train_orig[:n_show].to(device)
    lab_train = lab_train[:n_show].to(device)

    print(f"原始训练集样本形状: {x_train_orig.shape}")
    print(f"训练集条件数据形状: {lab_train.shape}")

    with torch.no_grad():
        gen_train = generate_samples(
            jit_model, lab_train, args, config, method=args.integrate_method
        )

    print(f"生成训练集样本形状: {gen_train.shape}")

    gen_train = gen_train.detach().cpu()
    x_train_orig = x_train_orig.detach().cpu()

    def denormalize_data_no_exp(norm_data, data_min, data_max):

        data_01 = (norm_data + 1) / 2
        data_log = data_01 * (data_max - data_min) + data_min
        return data_log

    original_samples_denorm = []
    for i in range(n_show):
        original_denorm = denormalize_data_no_exp(x_train_orig[i], data_min, data_max)
        original_samples_denorm.append(original_denorm)
        print(
            f"样本{i} 原始log(x+1)范围: [{original_denorm.min():.4f}, {original_denorm.max():.4f}]"
        )

    generated_samples_denorm = []
    for i in range(n_show):
        generated_denorm = denormalize_data_no_exp(gen_train[i], data_min, data_max)
        generated_samples_denorm.append(generated_denorm)
        print(
            f"样本{i} 生成log(x+1)范围: [{generated_denorm.min():.4f}, {generated_denorm.max():.4f}]"
        )

    print("\n=== 生成样本与原始样本误差统计 ===")
    for i in range(n_show):
        mse = torch.mean(
            (original_samples_denorm[i] - generated_samples_denorm[i]) ** 2
        )
        mae = torch.mean(
            torch.abs(original_samples_denorm[i] - generated_samples_denorm[i])
        )
        print(f"样本{i} - MSE: {mse.item():.6f}, MAE: {mae.item():.6f}")

    channel_idx = 0

    original_samples_ch0 = [sample[channel_idx] for sample in original_samples_denorm]
    generated_samples_ch0 = [sample[channel_idx] for sample in generated_samples_denorm]

    for i in range(n_show):
        print(
            f"样本{i} 第0通道原始log(x+1)范围: [{original_samples_ch0[i].min():.4f}, {original_samples_ch0[i].max():.4f}]"
        )
        print(
            f"样本{i} 第0通道生成log(x+1)范围: [{generated_samples_ch0[i].min():.4f}, {generated_samples_ch0[i].max():.4f}]"
        )

    sample_names = [f"Training Sample {i}" for i in range(n_show)]

    fig, axes = plt.subplots(n_show, 2, figsize=(12, 6 * n_show))

    for i in range(n_show):

        ax_orig = axes[i, 0] if n_show > 1 else axes[0]
        im_orig = ax_orig.imshow(original_samples_ch0[i], cmap="jet", vmin=4, vmax=8)
        ax_orig.set_title(f"{sample_names[i]} - Original (Layer 1)")
        plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)

        ax_gen = axes[i, 1] if n_show > 1 else axes[1]
        im_gen = ax_gen.imshow(generated_samples_ch0[i], cmap="jet", vmin=4, vmax=8)
        ax_gen.set_title(f"{sample_names[i]} - Generated (Layer 1)")
        plt.colorbar(im_gen, ax=ax_gen, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(rootpath, "comparison_layer1.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"可视化对比图已保存到: {save_path}")

    save_data = {
        "original_samples": torch.stack(original_samples_denorm).numpy(),
        "generated_samples": torch.stack(generated_samples_denorm).numpy(),
        "original_norm": x_train_orig.numpy(),
        "generated_norm": gen_train.numpy(),
    }

    with h5py.File(os.path.join(rootpath, "train_samples_comparison.h5"), "w") as f:
        f.create_dataset("original_samples", data=save_data["original_samples"])
        f.create_dataset("generated_samples", data=save_data["generated_samples"])
        f.create_dataset("original_norm", data=save_data["original_norm"])
        f.create_dataset("generated_norm", data=save_data["generated_norm"])

    print(f"对比数据已保存到: {os.path.join(rootpath, 'train_samples_comparison.h5')}")

    print("\n=== 生成真实样本（50次） ===")
    lab_test = torch.tensor(dobstrue_norm.reshape(1, -1), dtype=torch.float32).to(
        device
    )
    generated_data = []
    t3 = time.time()

    set_gen_seed(config.gen_seed)
    print(f"已固定生成50个样本的随机种子: {config.gen_seed}")

    for i in range(50):
        if i % 10 == 0:
            print(f"生成第 {i} 个真实样本...")
        with torch.no_grad():
            gen_sample = generate_samples(
                jit_model, lab_test, args, config, method=args.integrate_method
            )
        generated_data.append(gen_sample)

    generated_data = torch.stack(generated_data, dim=0).squeeze(1).detach().cpu()
    print(f"真实样本生成结果形状: {generated_data.shape}")

    gen_data_ = generated_data / 2 + 0.5
    gen_data1 = gen_data_ * (data_max - data_min) + data_min
    print(f"生成样本反归一化后范围: [{gen_data1.min():.4f}, {gen_data1.max():.4f}]")

    gen_data = torch.clamp(gen_data1, min=0.0)
    print(f"生成样本反归一化并截断后范围: [{gen_data.min():.4f}, {gen_data.max():.4f}]")

    gen_data_ex = np.exp(gen_data) - 1
    final_inv_para = gen_data_ex.reshape(gen_data_ex.shape[0], -1)
    print(f"生成样本展平后形状: {final_inv_para.shape}")

    save_path = "post_data.h5"
    with h5py.File(save_path, "w") as f:
        f.create_dataset("data", data=final_inv_para)
    print(f"生成的50个样本已保存为 {save_path} (形状: {final_inv_para.shape})")
    t4 = time.time()
    print(f"Sampling time: {t4-t3:.2f} seconds")

    return gen_data


def main():
    args = get_params()
    initial_res = initial(args)
    config = Config()
    if torch.cuda.is_available():
        print(
            f"\n初始GPU显存使用: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MiB"
        )

    mode = 1
    if mode == 0:

        print("=== 启动JiT模型训练（改进版） ===")
        rootpath = initial_res["rootpath"]
        os.chdir(rootpath)

        jit_model = JiT_MODELS[args.model_name](
            patch_size=args.patch_size,
            obs_dim=args.cdim,
            attn_drop=args.attn_drop,
            in_chans=config.channels,
            img_size=config.image_size,
        ).to(config.device)

        print(f"训练参数配置:")
        print(f"  - 时间步参数: P_mean={args.P_mean}, P_std={args.P_std}")
        print(f"  - 标签丢弃概率: {args.label_drop_prob} (使用固定特殊值0)")
        print(
            f"  - CFG参数: scale={args.cfg_scale}, interval=[{args.cfg_interval_min}, {args.cfg_interval_max}]"
        )

        t1 = time.time()
        train_losses, val_losses, best_val_loss = train_main(
            jit_model, initial_res["dataloader"], initial_res["testloader"], args
        )
        t2 = time.time()
        print(f"训练完成 | 总耗时: {t2 - t1:.2f}秒 | 最优验证损失: {best_val_loss:.6f}")

        loss_fig_path = os.path.join(args.output_dir, "train_val_loss.png")
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
        plt.plot(val_losses, label="Val Loss", color="red", linewidth=2)
        plt.title("Training & Validation Loss Curve", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
    elif mode == 1:

        print("=== 启动样本生成（改进版） ===")
        test_generate(initial_res, args, config)


if __name__ == "__main__":
    main()
