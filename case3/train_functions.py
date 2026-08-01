import torch
import torch.nn as nn
import math
import os
from tqdm import tqdm
from utils import EMA


class FlowMatcher:
    def __init__(
        self, P_mean=-0.8, P_std=0.8, noise_scale=1.0, t_eps=5e-2, label_drop_prob=0.15
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.label_drop_prob = label_drop_prob
        self.special_value = -2.0

        self.well_coords = torch.tensor(
            [
                [31, 31],
                [31, 95],
                [95, 31],
                [95, 95],
                [39, 39],
                [33, 29],
                [93, 90],
                [92, 35],
                [35, 99],
            ],
            dtype=torch.long,
        )

        self.well_loss_weight = 0

    def sample_t(self, batch_size, device):

        z = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def drop_labels(self, labels, is_training):

        if is_training and self.label_drop_prob > 0:
            drop_mask = (
                torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            )

            special_labels = torch.full_like(labels, self.special_value)
            return torch.where(drop_mask.unsqueeze(1), special_labels, labels)
        return labels

    def compute_well_constraint_loss(self, x_true, x_pred, device):

        well_coords = self.well_coords.to(device)
        B, C, H, W = x_true.shape
        well_loss = 0.0

        for b in range(B):
            for c in range(C):

                true_vals = x_true[b, c, well_coords[:, 0], well_coords[:, 1]]
                pred_vals = x_pred[b, c, well_coords[:, 0], well_coords[:, 1]]

                well_loss += torch.mean((true_vals - pred_vals) ** 2)

        well_loss = well_loss / (B * C)
        return well_loss

    def compute_loss(self, model, x, labels, is_training):

        batch_size = x.shape[0]
        device = x.device

        t = self.sample_t(batch_size, device).view(-1, 1, 1, 1)

        labels_dropped = self.drop_labels(labels, is_training)

        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = model(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        flow_loss = (v - v_pred).pow(2).mean()

        if is_training:
            well_loss = self.compute_well_constraint_loss(x, x_pred, device)

            total_loss = flow_loss + self.well_loss_weight * well_loss
        else:

            total_loss = flow_loss
            well_loss = torch.tensor(0.0, device=device)

        return total_loss, flow_loss, well_loss


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(model, dataloader, optimizer, epoch, args, ema, flow_matcher):
    model.train()
    total_loss = 0.0
    total_flow_loss = 0.0
    total_well_loss = 0.0
    pbar = tqdm(
        dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=args.rank != 0
    )

    for x, labels in pbar:
        x = x.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        loss, flow_loss, well_loss = flow_matcher.compute_loss(
            model, x, labels, is_training=True
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema.update(model)

        total_loss += loss.item()
        total_flow_loss += flow_loss.item()
        total_well_loss += well_loss.item()

        pbar.set_postfix(
            {
                "train_loss": loss.item(),
                "flow_loss": flow_loss.item(),
                "well_loss": well_loss.item(),
            }
        )

    avg_loss = total_loss / len(dataloader)
    avg_flow_loss = total_flow_loss / len(dataloader)
    avg_well_loss = total_well_loss / len(dataloader)
    return avg_loss, avg_flow_loss, avg_well_loss


def eval_one_epoch(model, testloader, args, flow_matcher):
    model.eval()
    total_val_loss = 0.0
    total_val_flow_loss = 0.0
    pbar = tqdm(testloader, desc=f"Val Epoch", disable=args.rank != 0)

    with torch.no_grad():
        for x, labels in pbar:
            x = x.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

            val_loss, val_flow_loss, _ = flow_matcher.compute_loss(
                model, x, labels, is_training=False
            )
            total_val_loss += val_loss.item()
            total_val_flow_loss += val_flow_loss.item()
            pbar.set_postfix(
                {"val_loss": val_loss.item(), "val_flow_loss": val_flow_loss.item()}
            )

    avg_val_loss = total_val_loss / len(testloader)
    avg_val_flow_loss = total_val_flow_loss / len(testloader)
    return avg_val_loss, avg_val_flow_loss


def save_model(
    model, ema, optimizer, epoch, loss, args, is_best=False, loss_type="val"
):
    if args.rank != 0:
        return

    if is_best:
        save_name = "model_best.pth"
        save_info = f"✅ 最优模型 | Epoch: {epoch + 1} | {loss_type}_loss: {loss:.6f}"
    else:
        save_name = "model_final.pth"
        save_info = (
            f"📌 最后一轮模型 | Epoch: {epoch + 1} | {loss_type}_loss: {loss:.6f}"
        )

    save_path = os.path.join(args.output_dir, save_name)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema_params,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "args": args,
        },
        save_path,
    )
    print(f"{save_info} | 保存路径: {save_path}")


def train_main(model, train_loader, val_loader, args):

    flow_matcher = FlowMatcher(
        P_mean=getattr(args, "P_mean", -0.8),
        P_std=getattr(args, "P_std", 0.8),
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        label_drop_prob=getattr(args, "label_drop_prob", 0.15),
    )

    if args.rank == 0:
        print(f"开始训练 | 总轮数: {args.epochs}")
        print(f"时间步参数: P_mean={flow_matcher.P_mean}, P_std={flow_matcher.P_std}")
        print(f"噪声尺度: {flow_matcher.noise_scale}, t_eps: {flow_matcher.t_eps}")
        print(
            f"标签丢弃概率: {flow_matcher.label_drop_prob}, 特殊值: {flow_matcher.special_value}"
        )
        print(
            f"井位数量: {len(flow_matcher.well_coords)}, 井位损失权重: {flow_matcher.well_loss_weight}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=args.ema_decay)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    train_flow_losses, train_well_losses = [], []

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_flow_loss, train_well_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, args, ema, flow_matcher
        )
        val_loss, val_flow_loss = eval_one_epoch(model, val_loader, args, flow_matcher)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_flow_losses.append(train_flow_loss)
        train_well_losses.append(train_well_loss)

        if args.rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train: {train_loss:.4f} (Flow: {train_flow_loss:.4f}, Well: {train_well_loss:.4f}) | "
                f"Val: {val_loss:.4f} (Flow: {val_flow_loss:.4f}) | "
                f"LR: {lr:.2e}"
            )

            if val_loss < best_val_loss and epoch >= 200:
                best_val_loss = val_loss
                save_model(model, ema, optimizer, epoch, val_loss, args, is_best=True)

    if args.rank == 0:
        save_model(
            model, ema, optimizer, args.epochs - 1, val_losses[-1], args, is_best=False
        )
        print(f"训练完成 | 最优验证损失: {best_val_loss:.4f}")

    return train_losses, val_losses, best_val_loss
