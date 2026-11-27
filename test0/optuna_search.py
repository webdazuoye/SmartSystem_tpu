import optuna
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import math
import subprocess

# 从你的 train.py 导入模型结构和种子固定函数
from train import set_seed, MLP


# 1. 定义目标函数
def objective(trial):
    # 1.1 定义搜索空间
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "hidden": trial.suggest_int("hidden", 32, 512, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }

    # 固定 Seed 为 0 和 30 个 Epoch
    seed = 0
    epochs = 30
    set_seed(seed)

    # 1.2 准备数据
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    idx = np.arange(len(y))
    rest, _ = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=0)

    train_dataset = TensorDataset(torch.tensor(X[train], dtype=torch.float32), torch.tensor(y[train], dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)

    # 1.3 初始化模型
    model = MLP(params["n_layers"], params["hidden"], params["dropout"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # 1.4 嵌套 MLflow 记录
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_param("seed", seed)

        val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_X = torch.tensor(X[val], dtype=torch.float32)
                val_y = torch.tensor(y[val], dtype=torch.long)
                val_outputs = model(val_X)
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
                val_acc = (val_preds == y[val]).mean()

            # 记录并汇报给 Optuna 的 Pruner
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            trial.report(val_acc, step=epoch)

            # 如果表现太差，提前停止这个 Trial 节省时间
            if trial.should_prune():
                mlflow.set_tag("state", "PRUNED")
                raise optuna.TrialPruned()

        mlflow.set_tag("state", "FAIL" if math.isnan(val_acc) else "COMPLETE")
        return val_acc


# 2. 主执行逻辑
if __name__ == "__main__":
    # 2.1 脏检查（如果代码没提交，直接报错，跟 train.py 逻辑一致）
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("Есть незакоммиченные изменения – запуск невозможен:\n" + dirty)

    # 2.2 设置 MLflow 和 Optuna
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")

    # 2.3 创建 Study (关键点：load_if_exists=True 实现断点续传)
    study = optuna.create_study(
        study_name="lr0_search",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    # 2.4 创建父级 MLflow Run
    with mlflow.start_run(run_name="lr0_search"):
        # 跑 40 次，使用 MaxTrialsCallback 保证中断后能续跑
        from optuna.study import MaxTrialsCallback

        study.optimize(
            objective,
            n_trials=40,
            callbacks=[MaxTrialsCallback(40, states=None)]
        )

        print("\n" + "=" * 50)
        print(f"最佳 Trial: {study.best_trial.number}")
        print(f"最佳参数: {study.best_params}")
        print("=" * 50 + "\n")