import os

import numpy as np
import polars as pl
import pyarrow as pa
import torch

def prepare_batch(batch_path, meta_path, sensor_geometry, training=True):
    meta_schema = pa.schema([
        ("event_id", pa.int64()),
        ("azimuth", pa.float32()),
        ("zenith", pa.float32()),
        ("batch_id", pa.int64())
    ])
    train_schema = pa.schema([
        ("time", pa.float32()),
        ("charge", pa.float32()),
        ("auxiliary", pa.bool_()),
        ("event_id", pa.int64()),
        ("sensor_id", pa.int16())
    ])
    
    df = pl.read_parquet(
        batch_path,
        use_pyarrow=True,
        #pyarrow_options={"schema": train_schema}
    )\
        .join(sensor_geometry, on="sensor_id", how="left")\
        .with_columns([
            (pl.col("time") - 1.0e04) / 3.0e4,
            pl.col("charge").log10() / 3.0,
            pl.col("auxiliary").cast(pl.Int16) + 1,  # reserve 0 for padding
            pl.col("sensor_id").cast(pl.Int16) + 1,
        ])\
        .groupby("event_id")\
        .agg([
            pl.col("sensor_id"),
            pl.col("time"),
            pl.col("charge"),
            pl.col("auxiliary"),
            pl.col("x"),
            pl.col("y"),
            pl.col("z")
        ])\
        .with_columns([
            pl.col("charge").arr.arg_max().alias("charge_max_idx")
        ])
    
    if training:
        batch_id = int(os.path.basename(batch_path).split(".")[0].split("_")[1])
        meta = pl.read_parquet(
            meta_path,
            columns=["event_id", "azimuth", "zenith"],
            use_pyarrow=True,
            pyarrow_options={
                "filters": [("batch_id", "==", batch_id)],
                #"schema": meta_schema,
            }
        )\
            .with_columns([
                (pl.col("azimuth").cos() * pl.col("zenith").sin()).alias("target_x"),
                (pl.col("azimuth").sin() * pl.col("zenith").sin()).alias("target_y"),
                (pl.col("zenith").cos()).alias("target_z")
            ])\
            #.drop(["azimuth", "zenith"])

        df = df.join(meta, on="event_id", how="left")

    return df
    
def collate_fn(seqs, min_idx, max_idx, padding_value, device, dtype):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seqs[i])[int(j.item()):int(k.item())] for i, (j, k) in enumerate(zip(min_idx, max_idx))],
        batch_first=True,
        padding_value=padding_value
    ).to(device).type(dtype)

def get_tensors(data, max_len, padding_value, device, dtype, training=True):
    charge_max_idx = torch.tensor(data["charge_max_idx"])
    min_idx = torch.max(torch.zeros(len(charge_max_idx)), charge_max_idx-max_len//2)
    max_idx = torch.max(min_idx+max_len, charge_max_idx+max_len//2)
    
    time = collate_fn(data["time"].to_numpy(), min_idx, max_idx, padding_value, device, dtype)
    auxiliary = collate_fn(data["auxiliary"].to_numpy(), min_idx, max_idx, padding_value, device, torch.int32)
    charge = collate_fn(data["charge"].to_numpy(), min_idx, max_idx, padding_value, device, dtype)
    xyz = torch.stack((
        collate_fn(data["x"].to_numpy(), min_idx, max_idx, padding_value=padding_value, device=device, dtype=dtype),
        collate_fn(data["y"].to_numpy(), min_idx, max_idx, padding_value=padding_value, device=device, dtype=dtype),
        collate_fn(data["z"].to_numpy(), min_idx, max_idx, padding_value=padding_value, device=device, dtype=dtype)
    )).permute(1, 2, 0)
    sensor_id = collate_fn(data["sensor_id"].to_numpy(), min_idx, max_idx, padding_value, device, torch.int32)
    inputs = {"time": time, "xyz": xyz, "auxiliary": auxiliary, "charge": charge, "sensor_id": sensor_id}
    if training:
        target = torch.tensor(data[["target_x", "target_y", "target_z"]].to_numpy(), dtype=dtype).to(device)
        azimuth = torch.tensor(data["azimuth"].to_numpy(), dtype=dtype).to(device)
        zenith = torch.tensor(data["zenith"].to_numpy(), dtype=dtype).to(device)
        return inputs, target, azimuth, zenith
    return inputs

"""
def xyz2azzen(x, y, z):
    # https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission
    r = np.sqrt(x**2 + y**2 + z**2)
    zenith = np.arccos(z/r)
    azimuth = np.arctan2(y, x) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    azimuth[azimuth < 0] = azimuth[azimuth < 0] + 2*np.pi
    return azimuth, zenith
"""

def xyz2azzen(x, y, z):
    # https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission
    r = torch.sqrt(x**2 + y**2 + z**2)
    zenith = torch.arccos(z/r)
    azimuth = torch.arctan2(y, x) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    azimuth[azimuth < 0] = azimuth[azimuth < 0] + 2*np.pi
    return azimuth, zenith
