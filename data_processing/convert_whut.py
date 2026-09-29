#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_whut.py —— WHUT-MSFVessel 三源数据解析

解析 scene 目录，把三源原始数据转成标准格式：
  1. AIS.txt            -> 解码 NMEA(AIVDM/AIVDO) -> ais_decoded.csv
  2. radar_track_gt.csv -> radar_targets.csv（含帧对应的时间戳）
  3. visible_track_gt.csv -> vision_targets.csv

依赖：pyais（pip install pyais）

用法：
  python convert_whut.py --scene H:/.../scene01 --out data/clean/

输出（统一字段，供 sync_multisource.py 后续对齐）：
  ais_decoded.csv    [timestamp_ms, mmsi, lat, lon, sog, cog]
  radar_targets.csv  [timestamp_ms, frame, track_id, x, y, w, h]
  vision_targets.csv [timestamp_ms, frame, track_id, x, y, w, h]
"""
import os
import csv
import glob
import argparse


# --------------------------------------------------------------------------
# AIS 解码（NMEA -> 结构化）
# --------------------------------------------------------------------------
def decode_ais(ais_path):
    """解析 AIS.txt，返回 [{ts_ms, mmsi, lat, lon, sog, cog}]"""
    from pyais import decode
    records = []
    with open(ais_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: <timestamp_ms>:<NMEA sentence>
            if ":" not in line:
                continue
            ts_str, sentence = line.split(":", 1)
            try:
                ts_ms = float(ts_str)
            except ValueError:
                continue
            sentence = sentence.strip()
            # 只处理 AIS 报文（AIVDM/AIVDO），跳过 GPGGA 等
            if not sentence.startswith("!"):
                continue
            if "AIVD" not in sentence:
                continue
            try:
                msg = decode(sentence.encode("ascii", errors="ignore"))
            except Exception:
                continue
            lat = getattr(msg, "lat", None)
            lon = getattr(msg, "lon", None)
            # 过滤无效坐标（缺省值会解出 91/181 这类越界值）
            if lat is None or lon is None:
                continue
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue
            records.append({
                "ts_ms": ts_ms,
                "mmsi": int(msg.mmsi),
                "lat": float(lat),
                "lon": float(lon),
                "sog": float(msg.speed) if msg.speed is not None else None,
                "cog": float(msg.course) if msg.course is not None else None,
            })
    records.sort(key=lambda r: r["ts_ms"])
    return records


# --------------------------------------------------------------------------
# 轨迹真值解析（MOT 格式：frame, id, x, y, w, h, -1, ...）
# --------------------------------------------------------------------------
def parse_mot_gt(gt_path):
    """解析 MOT 真值，返回 {frame: [{track_id, x, y, w, h}]}"""
    tracks = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            try:
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            except (ValueError, IndexError):
                continue
            tracks.setdefault(frame, []).append({"track_id": tid, "x": x, "y": y, "w": w, "h": h})
    return tracks


def list_image_times(img_dir):
    """从图像文件名提取时间戳（毫秒），按数值排序返回 [(ts_ms, filename)]"""
    pairs = []
    for fp in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        base = os.path.splitext(os.path.basename(fp))[0]
        # 视觉图文件名形如 2364716724_123.20_1.20_1.00，取第一段为时间戳
        ts_str = base.split("_")[0]
        try:
            pairs.append((float(ts_str), os.path.basename(fp)))
        except ValueError:
            continue
    pairs.sort(key=lambda p: p[0])
    return pairs


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="WHUT scene 目录（含 AIS.txt、radar_track_gt.csv、radar_images/ 等）")
    ap.add_argument("--out", default="data/clean", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene = args.scene

    # 1. AIS 解码
    ais_path = os.path.join(scene, "AIS.txt")
    if os.path.exists(ais_path):
        records = decode_ais(ais_path)
        with open(os.path.join(args.out, "ais_decoded.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "mmsi", "lat", "lon", "sog", "cog"])
            for r in records:
                w.writerow([round(r["ts_ms"], 0), r["mmsi"],
                            round(r["lat"], 6), round(r["lon"], 6),
                            round(r["sog"], 2) if r["sog"] is not None else "",
                            round(r["cog"], 1) if r["cog"] is not None else ""])
        mm = len({r["mmsi"] for r in records})
        print(f"AIS 解码 {len(records)} 条 / {mm} 艘船 -> {args.out}/ais_decoded.csv")

    # 2. 雷达轨迹真值 + 帧时间戳
    radar_gt = os.path.join(scene, "radar_track_gt.csv")
    radar_times = list_image_times(os.path.join(scene, "radar_images"))
    if os.path.exists(radar_gt):
        tracks = parse_mot_gt(radar_gt)
        with open(os.path.join(args.out, "radar_targets.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "frame", "track_id", "x", "y", "w", "h"])
            n = 0
            for frame in sorted(tracks):
                ts = radar_times[frame][0] if frame < len(radar_times) else ""
                for t in tracks[frame]:
                    w.writerow([ts, frame, t["track_id"], t["x"], t["y"], t["w"], t["h"]])
                    n += 1
        print(f"雷达目标 {n} 条 / {len(tracks)} 帧 -> {args.out}/radar_targets.csv")

    # 3. 视觉轨迹真值
    vis_gt = os.path.join(scene, "visible_track_gt.csv")
    vis_times = list_image_times(os.path.join(scene, "visible_images"))
    if os.path.exists(vis_gt):
        tracks = parse_mot_gt(vis_gt)
        with open(os.path.join(args.out, "vision_targets.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "frame", "track_id", "x", "y", "w", "h"])
            n = 0
            for frame in sorted(tracks):
                ts = vis_times[frame][0] if frame < len(vis_times) else ""
                for t in tracks[frame]:
                    w.writerow([ts, frame, t["track_id"], t["x"], t["y"], t["w"], t["h"]])
                    n += 1
        print(f"视觉目标 {n} 条 / {len(tracks)} 帧 -> {args.out}/vision_targets.csv")

    print(f"\n雷达帧 {len(radar_times)} / 视觉帧 {len(vis_times)}，解析完成。")


if __name__ == "__main__":
    main()
