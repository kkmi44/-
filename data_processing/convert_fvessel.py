#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_fvessel.py —— FVessel 数据集转换（AIS + 视频 + 相机参数）

把 FVessel 原始格式转换为本工程标准格式：
  - camera_para.txt -> 相机内参 K + 外参（保存为 calib.json）
  - ais/*.csv        -> 标准 AIS csv（timestamp,mmsi,lat,lon,sog,cog）
  - gt_fusion.txt    -> 融合真值（AIS-视觉关联标签，供第四章评测）

FVessel 原始格式（TITS 2023）：
  ais/2022_05_10_19_21_04.csv:
      [Number, MMSI, Lon, Lat, Speed, Course, Heading, Type, Timestamp(毫秒)]
  camera_para.txt:
      [Lon, Lat, 水平朝向, 垂直朝向, 相机高度, HFoV, VFoV, fx, fy, u0, v0]
  gt/Video-XX_gt_fusion.txt:
      <second>, <mmsi>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

用法：
  python convert_fvessel.py --video-dir <FVessel/01_Video+AIS/Video-XX目录> --out data/clean/

仅依赖标准库。
"""
import os
import csv
import json
import glob
import argparse


# --------------------------------------------------------------------------
# 相机参数
# --------------------------------------------------------------------------
def parse_camera_para(path):
    """解析 camera_para.txt -> dict（含内参 K 3x3 + 外参信息）"""
    with open(path, "r", encoding="utf-8") as f:
        vals = [float(x) for x in f.read().split()]
    # [Lon, Lat, 水平朝向, 垂直朝向, 高度, HFoV, VFoV, fx, fy, u0, v0]
    lon, lat, h_orient, v_orient, height, hfov, vfov, fx, fy, u0, v0 = vals[:12]
    K = [[fx, 0, u0], [0, fy, v0], [0, 0, 1]]
    calib = {
        "position": {"lat": lat, "lon": lon, "height_m": height},
        "orientation": {"horizontal_deg": h_orient, "vertical_deg": v_orient},
        "fov": {"horizontal_deg": hfov, "vertical_deg": vfov},
        "intrinsic_K": K,
    }
    return calib


# --------------------------------------------------------------------------
# AIS 解析（注意 FVessel 列序：MMSI, Lon, Lat, Speed, Course, ... Timestamp 毫秒）
# --------------------------------------------------------------------------
def parse_ais_dir(ais_dir):
    """解析 ais/ 目录下所有 csv，合并去重，返回 [{ts(秒), mmsi, lat, lon, sog, cog}]"""
    records = []
    for fp in sorted(glob.glob(os.path.join(ais_dir, "*.csv"))):
        with open(fp, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            # 自动定位列
            h = [c.strip().lower() for c in header]
            def idx(*names):
                for n in names:
                    if n in h:
                        return h.index(n)
                return None
            i_mmsi, i_lon, i_lat = idx("mmsi"), idx("lon", "lng", "longitude"), idx("lat", "latitude")
            i_speed, i_course, i_ts = idx("speed", "sog"), idx("course", "cog", "heading"), idx("timestamp", "time", "ts")
            if i_mmsi is None or i_ts is None:
                continue
            for row in reader:
                if not row:
                    continue
                try:
                    ts_raw = float(row[i_ts])
                    ts = ts_raw / 1000.0 if ts_raw > 1e12 else ts_raw  # 毫秒转秒
                    rec = {
                        "ts": ts,
                        "mmsi": int(float(row[i_mmsi])),
                        "lat": float(row[i_lat]) if i_lat is not None else None,
                        "lon": float(row[i_lon]) if i_lon is not None else None,
                        "sog": float(row[i_speed]) if (i_speed is not None and row[i_speed] not in ("", "-1")) else None,
                        "cog": float(row[i_course]) if (i_course is not None and row[i_course] not in ("", "-1")) else None,
                    }
                    if rec["lat"] is not None and rec["lon"] is not None:
                        records.append(rec)
                except (ValueError, IndexError):
                    continue
    # 按 (mmsi, ts) 去重
    records.sort(key=lambda r: (r["mmsi"], r["ts"]))
    dedup = []
    last = None
    for r in records:
        if last and abs(r["ts"] - last["ts"]) < 1e-3 and r["mmsi"] == last["mmsi"]:
            continue
        dedup.append(r)
        last = r
    return dedup


# --------------------------------------------------------------------------
# 融合真值（gt_fusion.txt）
# --------------------------------------------------------------------------
def parse_gt_fusion(path):
    """解析 <second>,<mmsi>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "second": float(parts[0]),
                    "mmsi": int(float(parts[1])),
                    "bb_left": float(parts[2]), "bb_top": float(parts[3]),
                    "bb_width": float(parts[4]), "bb_height": float(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    return rows


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="FVessel 单个视频目录（含 ais/、camera_para.txt、gt/）")
    ap.add_argument("--out", default="data/clean", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vd = args.video_dir

    # 相机参数
    cam_path = os.path.join(vd, "camera_para.txt")
    if os.path.exists(cam_path):
        calib = parse_camera_para(cam_path)
        with open(os.path.join(args.out, "calib.json"), "w", encoding="utf-8") as f:
            json.dump(calib, f, ensure_ascii=False, indent=2)
        print(f"相机参数 -> {args.out}/calib.json")

    # AIS
    ais_dir = os.path.join(vd, "ais")
    if os.path.isdir(ais_dir):
        records = parse_ais_dir(ais_dir)
        with open(os.path.join(args.out, "ais_fvessel.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "mmsi", "lat", "lon", "sog", "cog"])
            for r in records:
                w.writerow([round(r["ts"], 3), r["mmsi"],
                            round(r["lat"], 6) if r["lat"] is not None else "",
                            round(r["lon"], 6) if r["lon"] is not None else "",
                            round(r["sog"], 2) if r["sog"] is not None else "",
                            round(r["cog"], 1) if r["cog"] is not None else ""])
        print(f"AIS {len(records)} 条 -> {args.out}/ais_fvessel.csv")

    # 融合真值
    gt_dir = os.path.join(vd, "gt")
    if os.path.isdir(gt_dir):
        gt_files = glob.glob(os.path.join(gt_dir, "*_fusion.txt"))
        all_gt = []
        for gf in gt_files:
            all_gt.extend(parse_gt_fusion(gf))
        with open(os.path.join(args.out, "gt_fusion.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["second", "mmsi", "bb_left", "bb_top", "bb_width", "bb_height"])
            for g in all_gt:
                w.writerow([g["second"], g["mmsi"], g["bb_left"], g["bb_top"],
                            g["bb_width"], g["bb_height"]])
        print(f"融合真值 {len(all_gt)} 条 -> {args.out}/gt_fusion.csv")

    print("\n转换完成。")


if __name__ == "__main__":
    main()
