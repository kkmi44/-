#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sync_multisource.py —— 多源时空同步（论文 4.2.2）

以「雷达采样时刻」为系统主时钟，把 AIS、视觉对齐到同一时间基准：

  - AIS 线性插值对齐（公式 4-7）:
        fxa,k = fxa,1 + (t_r,k − t_a,1)/(t_a,2 − t_a,1) · (fxa,2 − fxa,1)
    状态向量 fxa = [lat, lon, SOG, COG]，其中 COG 用最短角差插值避免 360° 跳变。
  - 视觉帧：取时间最近的帧（帧率一致时即同帧）。

用法：
  # 雷达时刻来自文件（每行一个时间戳），AIS 已清洗，视觉框已检测
  python sync_multisource.py \
      --radar-times radar_times.txt \
      --ais data/clean/ais_clean.csv \
      --vision data/clean/vision_boxes.csv \
      --out data/sync/sync_frames.csv

仅依赖标准库。
"""
import csv
import math
import argparse
from collections import defaultdict


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def shortest_angle_diff(a, b):
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def angle_lerp(a, b, t):
    return (a + shortest_angle_diff(a, b) * t) % 360.0


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
_ALIAS = {
    "lat": ["lat", "latitude"], "lon": ["lon", "lng", "long", "longitude"],
    "sog": ["sog", "speed"], "cog": ["cog", "course", "heading"],
    "ts": ["timestamp", "time", "ts"], "mmsi": ["mmsi", "shipid", "id"],
    "u": ["u", "cx"], "v": ["v", "cy"], "w": ["w", "width"], "h": ["h", "height"],
    "conf": ["conf", "score"],
}


def detect_columns(header):
    h = [str(c).strip().lower() for c in header]
    m = {}
    for std, names in _ALIAS.items():
        for i, c in enumerate(h):
            if c in names and std not in m:
                m[std] = i
                break
    return m


def read_ais(path):
    """读取 AIS，返回 {mmsi: [按时间排序的 dict 列表]}"""
    tracks = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = detect_columns(header)
        def g(row, k):
            i = cols.get(k)
            return float(row[i]) if (i is not None and row[i] not in ("", "-1")) else None
        for row in reader:
            if not row or row[0].strip() == "":
                continue
            lat, lon, ts = g(row, "lat"), g(row, "lon"), g(row, "ts")
            if lat is None or lon is None or ts is None:
                continue
            tracks[g(row, "mmsi")].append({
                "ts": ts, "lat": lat, "lon": lon,
                "sog": g(row, "sog"), "cog": g(row, "cog"),
            })
    for k in tracks:
        tracks[k].sort(key=lambda r: r["ts"])
    return tracks


def read_vision(path):
    """读取视觉框，返回按时间排序的 [{ts, u, v, w, h, conf}]"""
    boxes = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = detect_columns(header)
        def g(row, k):
            i = cols.get(k)
            return float(row[i]) if (i is not None and row[i] not in ("", "-1")) else None
        for row in reader:
            if not row or row[0].strip() == "":
                continue
            ts = g(row, "ts")
            if ts is None:
                continue
            boxes.append({"ts": ts, "u": g(row, "u"), "v": g(row, "v"),
                          "w": g(row, "w"), "h": g(row, "h"), "conf": g(row, "conf")})
    boxes.sort(key=lambda b: b["ts"])
    return boxes


def read_radar_times(path):
    """雷达帧时间戳：文件每行一个；或从目录内文件名解析（统一按数值排序）"""
    import os
    times = []
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            base = os.path.splitext(fn)[0]
            try:
                times.append(float(base))
            except ValueError:
                pass
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))
    return sorted(times)


# --------------------------------------------------------------------------
# 核心：插值与同步
# --------------------------------------------------------------------------
def interpolate_ais(track, t_r):
    """单船轨迹在 t_r 时刻线性插值（公式 4-7），返回 dict 或 None"""
    if not track:
        return None
    # 找 t_r 前后的最近两点
    if t_r <= track[0]["ts"]:
        return track[0]
    if t_r >= track[-1]["ts"]:
        return track[-1]
    for i in range(len(track) - 1):
        a, b = track[i], track[i + 1]
        if a["ts"] <= t_r <= b["ts"]:
            if b["ts"] == a["ts"]:
                return a
            t = (t_r - a["ts"]) / (b["ts"] - a["ts"])
            def lerp(x, y): return None if (x is None or y is None) else x + (y - x) * t
            cog = angle_lerp(a["cog"] or 0, b["cog"] or 0, t) \
                if (a["cog"] is not None and b["cog"] is not None) else None
            return {"ts": t_r, "lat": lerp(a["lat"], b["lat"]),
                    "lon": lerp(a["lon"], b["lon"]),
                    "sog": lerp(a["sog"], b["sog"]), "cog": cog}
    return None


def nearest_vision(boxes, t_r, tol):
    """找 t_r 附近最近的一帧视觉框集合，超过 tol 秒返回空"""
    if not boxes:
        return []
    # 二分找最近
    import bisect
    ts_list = [b["ts"] for b in boxes]
    idx = bisect.bisect_left(ts_list, t_r)
    cands = []
    if idx < len(boxes):
        cands.append(boxes[idx])
    if idx > 0:
        cands.append(boxes[idx - 1])
    best = min(cands, key=lambda b: abs(b["ts"] - t_r))
    if abs(best["ts"] - t_r) > tol:
        return []
    return [b for b in boxes if abs(b["ts"] - best["ts"]) <= 1e-9]


def sync(radar_times, ais_tracks, vision_boxes, tol=1.0):
    """主时钟同步，返回对齐观测列表"""
    frames = []
    for t_r in radar_times:
        ais_aligned = []
        for mmsi, track in ais_tracks.items():
            r = interpolate_ais(track, t_r)
            if r is not None:
                r = dict(r)
                r["mmsi"] = mmsi
                ais_aligned.append(r)
        vis = nearest_vision(vision_boxes, t_r, tol)
        frames.append({
            "ts": t_r,
            "ais": ais_aligned,
            "vision": vis,
        })
    return frames


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="多源时空同步（论文4.2.2）")
    ap.add_argument("--radar-times", required=True, help="雷达帧时间戳文件或雷达图目录")
    ap.add_argument("--ais", required=True, help="AIS 轨迹 csv（已清洗）")
    ap.add_argument("--vision", default=None, help="视觉框 csv（带时间戳）")
    ap.add_argument("--tol", type=float, default=1.0, help="视觉帧对齐容差(秒)")
    ap.add_argument("--out", default="sync_frames.csv", help="输出路径")
    args = ap.parse_args()

    radar_times = read_radar_times(args.radar_times)
    ais_tracks = read_ais(args.ais)
    vision_boxes = read_vision(args.vision) if args.vision else []

    print(f"雷达帧 {len(radar_times)} 个 / AIS 轨迹 {len(ais_tracks)} 艘 / 视觉框 {len(vision_boxes)} 条")

    frames = sync(radar_times, ais_tracks, vision_boxes, args.tol)

    # 输出：每帧一行，含对齐的 AIS 目标数、视觉框数，以及每船的插值状态
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "mmsi", "lat", "lon", "sog", "cog", "n_vision"])
        for fr in frames:
            n_vis = len(fr["vision"])
            for a in fr["ais"]:
                w.writerow([round(fr["ts"], 3), a["mmsi"],
                            round(a["lat"], 6), round(a["lon"], 6),
                            round(a["sog"], 2) if a["sog"] is not None else "",
                            round(a["cog"], 1) if a["cog"] is not None else "",
                            n_vis])
            if not fr["ais"]:
                w.writerow([round(fr["ts"], 3), "", "", "", "", "", n_vis])

    aligned = sum(1 for fr in frames if fr["ais"])
    print(f"\n同步完成: {len(frames)} 帧，其中 {aligned} 帧有对齐 AIS 目标")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
