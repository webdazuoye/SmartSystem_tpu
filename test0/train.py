import argparse
import yaml
import os
import subprocess
import json
import hashlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import mlflow
import time
import uuid


# --------- 1. 辅助函数 ---------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def check_git_status():
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("Есть незакоммиченные изменения – итоговый запуск невозможен:\n" + dirty)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return commit


def get_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------- 2. 模型定义 ---------
class MLP(nn.Module):
    def __init__(self, n_layers, hidden, dropout):
        super().__init__()
        layers = []
        in_dim = 64
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------- 3. 主函数 ---------
def main(args):
    # 3.1 加载与合并配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config['seed'] = args.seed
    if args.eval_test:
        config['eval_test'] = True

    # 3.2 Git检查
    commit_hash = check_git_status()

    # 3.3 设置运行目录
    run_id = str(uuid.uuid4())[:8]
    run_dir = f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    # 3.4 准备数据
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # 归一化
    idx = np.arange(len(y))

    # 严格按老师给的 random_state=0 划分
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=0)

    split_data = {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
    with open("data/split.json", 'w', encoding='utf-8') as f:
        json.dump(split_data, f)

    split_hash = get_sha256("data/split.json")

    # 3.5 数据加载器
    set_seed(config['seed'])
    torch.set_num_threads(config['runtime']['num_threads'])

    train_dataset = TensorDataset(torch.tensor(X[train], dtype=torch.float32), torch.tensor(y[train], dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)

    # 3.6 初始化模型
    model = MLP(config['model']['n_layers'], config['model']['hidden'], config['model']['dropout'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

    # 3.7 记录 metadata
    metadata = {
        "run_id": run_id,
        "status": "running",
        "commit": commit_hash,
        "seed": config['seed'],
        "split_hash": split_hash,
        "start_time": time.time()
    }

    # 3.8 MLflow 记录
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")

    with mlflow.start_run(run_name=f"run_{run_id}"):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['train'])
        mlflow.log_param("seed", config['seed'])

        history = []

        # 3.9 训练循环
        for epoch in range(config['train']['epochs']):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train)

            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_X = torch.tensor(X[val], dtype=torch.float32)
                val_y = torch.tensor(y[val], dtype=torch.long)
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs, val_y).item()
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
                val_acc = (val_preds == y[val]).mean()

            # 记录指标
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

            print(
                f"Epoch {epoch + 1}/{config['train']['epochs']} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        # 3.10 保存结果
        pd.DataFrame(history).to_csv(f"{run_dir}/history.csv", index=False)

        predictions = [{"split": "val", "index": i, "y_true": true, "y_pred": pred} for i, true, pred in
                       zip(val, y[val], val_preds)]

        if config.get('eval_test'):
            with torch.no_grad():
                test_X = torch.tensor(X[test], dtype=torch.float32)
                test_outputs = model(test_X)
                test_preds = torch.argmax(test_outputs, dim=1).numpy()
                predictions.extend([{"split": "test", "index": i, "y_true": true, "y_pred": pred} for i, true, pred in
                                    zip(test, y[test], test_preds)])

        pd.DataFrame(predictions).to_csv(f"{run_dir}/predictions.csv", index=False)

        metrics = {"val_acc": val_acc, "val_loss": val_loss}
        with open(f"{run_dir}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f)

        metadata["status"] = "completed"
        metadata["end_time"] = time.time()
        with open(f"{run_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f)

        mlflow.log_artifacts(run_dir, artifact_path="run_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()
    main(args)