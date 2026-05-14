import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from transformers import TrainerCallback


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GPUStatsLogger:
    def __init__(self, device_index=0):
        self.device_index = device_index

    def snapshot(self):
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if torch.cuda.is_available():
            device_index = self.device_index if self.device_index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            total_mb = props.total_memory / 1024 / 1024
            allocated_mb = torch.cuda.memory_allocated(device_index) / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved(device_index) / 1024 / 1024
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            free_mb = free_bytes / 1024 / 1024
            total_visible_mb = total_bytes / 1024 / 1024
            used_mb = total_visible_mb - free_mb
            gpu_util_pct = round((used_mb / total_visible_mb) * 100, 2) if total_visible_mb else None
            vram_util_pct = round((allocated_mb / total_mb) * 100, 2) if total_mb else None

            data.update(
                {
                    "device_index": device_index,
                    "vram_total_mb": round(total_mb, 2),
                    "vram_allocated_mb": round(allocated_mb, 2),
                    "vram_reserved_mb": round(reserved_mb, 2),
                    "vram_free_mb": round(free_mb, 2),
                    "vram_total_visible_mb": round(total_visible_mb, 2),
                    "gpu_util_pct": gpu_util_pct,
                    "vram_util_pct": vram_util_pct,
                }
            )

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        data.update(
            {
                "rss_mb": round(mem.rss / 1024 / 1024, 2),
                "vms_mb": round(mem.vms / 1024 / 1024, 2),
            }
        )
        return data

    def line(self, prefix="GPU"):
        snap = self.snapshot()
        parts = [prefix]
        if "gpu_util_pct" in snap and snap["gpu_util_pct"] is not None:
            parts.append(f'GPU {snap["gpu_util_pct"]}%')
        if "vram_util_pct" in snap and snap["vram_util_pct"] is not None:
            parts.append(f'VRAM {snap["vram_util_pct"]}%')
        if "vram_allocated_mb" in snap and "vram_total_mb" in snap:
            parts.append(f'{snap["vram_allocated_mb"]}/{snap["vram_total_mb"]} MB')
        return " | ".join(parts)

    def print(self, prefix="GPU"):
        print(self.line(prefix=prefix))


class HistoryStore:
    def __init__(self, output_dir, filename_jsonl="history.jsonl", filename_csv="history.csv"):
        self.output_dir = ensure_dir(output_dir)
        self.path_jsonl = self.output_dir / filename_jsonl
        self.path_csv = self.output_dir / filename_csv
        self.rows = []

    def add(self, row):
        row = dict(row)
        self.rows.append(row)
        with self.path_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_csv(self):
        if not self.rows:
            return self.path_csv
        keys = []
        for row in self.rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        with self.path_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.path_csv

    def to_frame(self):
        return pd.DataFrame(self.rows)

    def last(self):
        if not self.rows:
            return None
        return self.rows[-1]


class GPUAndHistoryCallback(TrainerCallback):
    def __init__(self, history_store, gpu_logger=None, every_n_logs=1):
        self.history_store = history_store
        self.gpu_logger = gpu_logger if gpu_logger is not None else GPUStatsLogger()
        self.every_n_logs = every_n_logs
        self._log_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        self._log_count += 1
        if self._log_count % self.every_n_logs != 0:
            return control

        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": int(state.global_step),
        }
        row.update(logs)
        row.update(self.gpu_logger.snapshot())
        self.history_store.add(row)

        loss = row.get("loss", None)
        eval_loss = row.get("eval_loss", None)
        gpu_util_pct = row.get("gpu_util_pct", None)
        vram_util_pct = row.get("vram_util_pct", None)

        msg = f"step {state.global_step}"
        if loss is not None:
            msg += f" | loss {loss:.4f}"
        if eval_loss is not None:
            msg += f" | eval_loss {eval_loss:.4f}"
        if gpu_util_pct is not None:
            msg += f" | GPU {gpu_util_pct}%"
        if vram_util_pct is not None:
            msg += f" | VRAM {vram_util_pct}%"

        print(msg)
        return control


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def save_trainer_history(trainer, output_dir):
    output_dir = ensure_dir(output_dir)
    history_path = output_dir / "trainer_state_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    return history_path


def token_length_stats(lengths):
    arr = np.asarray(lengths)

    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def print_cuda_rows():
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    n = torch.cuda.device_count()

    for idx in range(n):
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / 1024**3
        print(
            f"GPU {idx} | "
            f"{props.name} | "
            f"{total:.2f} GB VRAM | "
            f"cc {props.major}.{props.minor}"
        )


def plot_histogram(
    values,
    title,
    xlabel,
    ylabel,
    bins=50,
    path=None,
    figsize=(10, 4),
):
    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=160)

    plt.grid(alpha=0.5)
    plt.show()


def plot_lines(
    df,
    x,
    ys,
    title,
    xlabel,
    ylabel,
    path=None,
    figsize=(10, 4),
    log_y=False,
):
    plt.figure(figsize=figsize)

    for col in ys:
        if col not in df.columns:
            continue

        series = df[[x, col]].dropna()

        if series.empty:
            continue

        if log_y:
            plt.semilogy(series[x], series[col], label=col)
        else:
            plt.plot(series[x], series[col], label=col)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=160)

    plt.grid(alpha=0.5)
    plt.show()