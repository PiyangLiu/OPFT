import torch
import numpy as np
import h5py
import os
import logging
import torch.distributed as dist


def normalize_data(data, data_min, data_max):

    data_norm = (data - data_min) / (data_max - data_min)
    data_norm = data_norm * 2 - 1
    return data_norm


def denormalize_data(data_norm, data_min, data_max):

    data = (data_norm + 1) / 2
    data = data * (data_max - data_min) + data_min
    return data


def load_h5_data(file_path, key_x="x", key_y="y"):

    with h5py.File(file_path, "r") as f:
        x = np.array(f[key_x])
        y = np.array(f[key_y])

    x = x.reshape(-1, 139, 48, 9).transpose(0, 3, 1, 2)
    return x, y


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.gpu)
    args.distributed = True


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("oil_reservoir_inversion")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


class EMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = torch.nn.ModuleList(model.children())
        self.decay = decay
        self.ema_params = [p.detach().clone() for p in model.parameters()]

    def update(self, model):
        for ema_p, p in zip(self.ema_params, model.parameters()):
            ema_p.data = ema_p.data * self.decay + p.data * (1 - self.decay)

    def apply(self):
        for ema_p, p in zip(self.ema_params, self.ema_model.parameters()):
            p.data = ema_p.data.clone()
