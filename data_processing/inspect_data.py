#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_data.py —— 数据格式探查工具

在虚拟机上对下载好的 WHUT/FVessel 数据跑一遍，把目录结构、CSV 表头、
文件命名等信息打印出来。把输出发给我，我据此写准确的转换脚本。

用法：
  python inspect_data.py <数据目录或文件> [--depth 2] [--csv-lines 5]

仅依赖标准库。
"""
import os
import csv
import argparse


def walk_tree(root, depth):
    """打印目录结构（前 depth 层）"""
    print(f"\n=== 目录结构: {root} ===")
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        level = 0 if rel == "." else rel.count(os.sep) + 1
        if level > depth:
            dirnames[:] = []
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath) or rel}/")
        for fn in sorted(filenames)[:10]:
            fp = os.path.join(dirpath, fn)
            size = os.path.getsize(fp)
            print(f"{indent}  {fn}  ({size} B)")
        if len(filenames) > 10:
            print(f"{indent}  ... 共 {len(filenames)} 个文件")
        if level >= depth:
            dirnames[:] = []


def show_csv(path, n):
    """打印 CSV 前 n 行"""
    print(f"\n=== CSV: {path} ===")
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                print(" | ".join(row))
                if i >= n:
                    break
    except Exception as e:
        print(f"[读取失败] {e}")


def show_text(path, n):
    """打印文本文件前 n 行"""
    print(f"\n=== 文本: {path} ===")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                print(line.rstrip())
                if i >= n:
                    break
    except Exception as e:
        print(f"[读取失败] {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="数据目录或文件")
    ap.add_argument("--depth", type=int, default=2, help="目录深度")
    ap.add_argument("--csv-lines", type=int, default=5, help="CSV/文本预览行数")
    args = ap.parse_args()

    p = args.path
    if os.path.isdir(p):
        walk_tree(p, args.depth)
        # 自动预览目录下的 csv/txt（最多 3 个）
        cnt = 0
        for dirpath, _, filenames in os.walk(p):
            for fn in sorted(filenames):
                fp = os.path.join(dirpath, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext in (".csv",) and cnt < 3:
                    show_csv(fp, args.csv_lines)
                    cnt += 1
                elif ext in (".txt",) and not fn.startswith(".") and cnt < 3:
                    show_text(fp, args.csv_lines)
                    cnt += 1
                if cnt >= 3:
                    break
    elif os.path.isfile(p):
        ext = os.path.splitext(p)[1].lower()
        if ext == ".csv":
            show_csv(p, args.csv_lines)
        else:
            show_text(p, args.csv_lines)
    else:
        print(f"路径不存在: {p}")


if __name__ == "__main__":
    main()
