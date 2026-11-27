import os
import json
import glob
import pandas as pd

# 找到最近的 10 个 run 文件夹（按修改时间排序）
runs = sorted(glob.glob('runs/*'), key=os.path.getmtime)[-10:]

results = []
for run in runs:
    metrics_path = os.path.join(run, 'metrics.json')
    config_path = os.path.join(run, 'config.yaml')
    predictions_path = os.path.join(run, 'predictions.csv')

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # 读 config 判断是 base 还是 best
    with open(config_path, 'r') as f:
        config_text = f.read()
        conf_type = "best" if "491" in config_text else "base"  # 491是之前找出的最佳 hidden 节点数

    # 从 predictions.csv 现算 test_acc (因为我们之前的 metrics.json 漏存了)
    test_acc = 0.0
    if os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
        test_df = df[df['split'] == 'test']
        if not test_df.empty:
            test_acc = (test_df['y_true'] == test_df['y_pred']).mean()

    results.append({
        "run_id": os.path.basename(run),
        "type": conf_type,
        "val_acc": metrics.get("val_acc", 0),
        "test_acc": test_acc
    })

# 格式化输出
print(f"{'Type':<6} | {'Val Acc':<10} | {'Test Acc':<10}")
print("-" * 32)
for r in results:
    print(f"{r['type']:<6} | {r['val_acc']:.4f}     | {r['test_acc']:.4f}")

print("\n--- 计算差值 (Best - Base) ---")
# 按顺序每两个是一对 (base, best)
for i in range(0, len(results), 2):
    base = results[i]
    best = results[i + 1]
    # 确认一下顺序，保证 base 在前，best 在后
    if base['type'] == 'best':
        base, best = best, base

    delta_test = best['test_acc'] - base['test_acc']
    print(f"Seed {101 + i // 2}: Base={base['test_acc']:.4f}, Best={best['test_acc']:.4f}, Delta={delta_test:+.4f}")