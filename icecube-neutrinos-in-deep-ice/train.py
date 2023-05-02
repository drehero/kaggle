import gc
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import polars as pl
import torch

from config import config24 as config
from data import *
from loss import *
from model import Net


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in config.device else "cpu" # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data
np.random.seed(config.seed)
batches = np.sort(os.listdir(os.path.join(config.in_dir, "train")))
val_batches = np.random.choice(batches, 60, replace=False)
train_batches = [b for b in batches if b not in val_batches]
sens_geo_dtypes = {
    "sensor_id": pl.Int16,
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,
}
sensor_geometry = pl.read_csv(
    os.path.join(config.in_dir, "sensor_geometry.csv"),
    dtypes=sens_geo_dtypes
)\
    .with_columns([
        pl.col("x") / 500,
        pl.col("y") / 500,
        pl.col("z") / 500,
    ])
meta_path = os.path.join(config.in_dir, "train_meta.parquet")

# model init
n_iter = 0
best_metric = 1e9
seen_batches = []

model = Net(config)
if config.resume_training:
    ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    n_iter = checkpoint["n_iter"]
    best_metric = checkpoint["best_metric"]
    seen_batches = checkpoint["seen_batches"]
model.to(config.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))
# optimizer
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
if config.resume_training:
    optimizer.load_state_dict(checkpoint["optimizer"])
                        
# logging
if config.wandb_log:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)

t0 = time.time()
losses = []
metrics = []

for fn in train_batches:
    if n_iter > config.max_iters:
        break
    if fn not in seen_batches:
        seen_batches += [fn]
        batch_path = os.path.join(config.in_dir, "train", fn)
        print(batch_path)
 
        df = prepare_batch(batch_path, meta_path, sensor_geometry)

        for data in df.iter_slices(config.batch_size):
            inputs, target, az_true, zen_true = get_tensors(
                data, max_len=config.max_len, padding_value=config.padding_value, device=config.device, dtype=ptdtype, training=True
            )

            # determine and set the learning rate for this iteration
            if config.decay_lr:
                lr = get_lr(n_iter, config.learning_rate, config.min_lr, config.warmup_iters, config.lr_decay_iters)
            else:
                lr = config.learning_rate
                
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                
            with ctx:
                out = model(inputs)
                loss = vMF_loss(out, target)
                #loss = F.mse_loss(out, target)
                #az_pred, zen_pred = xyz2azzen(out[:, 0], out[:, 1], out[:, 2])
                #loss = angular_dist_loss(
                #    az_pred=out[:, 0], zen_pred=out[:, 1], az_true=az_true, zen_true=zen_true
                #)
                
            losses += [loss.item()]
            
            scaler.scale(loss).backward()
            
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            
            # keep track of metric
            with torch.no_grad():
                az_pred, zen_pred = xyz2azzen(out[:, 0], out[:, 1], out[:, 2])
                metric = angular_dist_score(
                    az_pred, zen_pred, az_true=az_true, zen_true=zen_true
                )
            
            metrics += [metric.item()]
                
            if n_iter % config.eval_interval == 0:
                mean_loss = np.mean(losses)
                mean_metric = np.mean(metrics)
                print(f"step {n_iter}: metric {mean_metric:.4f}, loss {mean_loss:.4f}")
                if config.wandb_log:
                    wandb.log({
                        "iter": n_iter,
                        "loss": mean_loss,
                        "metric": mean_metric,
                        "lr": lr
                    })
                if mean_metric < best_metric or config.always_save_checkpoint:
                    best_metric = mean_metric
                    if n_iter > 0:
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "n_iter": n_iter,
                            "best_metric": best_metric,
                            "config": config,
                            "seen_batches": seen_batches
                        }
                        print(f"saving checkpoint to {config.out_dir}")
                        torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt")) 
                losses = []
                metrics = []
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if n_iter % config.log_interval == 0:
                lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
                metricf = metric.item()
                print(f"iter {n_iter}: metric {metricf:.4f}, loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
            n_iter += 1
            
            if n_iter > config.max_iters:
                break

        del(df)
        gc.collect()
