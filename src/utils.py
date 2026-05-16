"""工具函数: 日志、路径、配置保存等。"""

import os
import json
import yaml
import subprocess
from datetime import datetime


def get_timestamp():
    """返回当前时间戳字符串，用于 run 目录命名。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(experiment_name):
    """创建 run 目录: runs/emotionroi/{experiment_name}_{timestamp}/"""
    timestamp = get_timestamp()
    run_dir = os.path.join("runs", "emotionroi", f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(config, run_dir):
    """将配置保存为 config.yaml。"""
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_class_to_idx(class_names, run_dir):
    """保存 class_to_idx 映射为 JSON。"""
    path = os.path.join(run_dir, "class_to_idx.json")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)


def save_git_info(run_dir):
    """保存当前 git commit 和 branch 信息。"""
    path = os.path.join(run_dir, "git_info.txt")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        with open(path, "w") as f:
            f.write(f"commit: {commit}\nbranch: {branch}\n")
    except Exception:
        with open(path, "w") as f:
            f.write("git info unavailable\n")


class Logger:
    """同时输出到控制台和日志文件的 Logger。"""

    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8")

    def log(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()
