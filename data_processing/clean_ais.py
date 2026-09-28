#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clean_ais.py —— AIS/ARPA 轨迹数据清洗

清洗流程（对齐论文 3.1 节的预处理标准）：
  1. 自动识别列名，读入轨迹
  2. 按时间戳排序、去重
  3. 异常检测：位置跳变 / 航速越界 / 航向越界
  4. 可选：线性插值到等间隔时间网格（航向用最短角差插值避免 360° 跳变）
  5. 输出清洗后 CSV + 清洗报告

用法：
  python clean_ais.py -i data.csv -o clean_data.csv
  python clean_ais.py -i data.csv -o clean_data.csv --interpolate 5 --max-speed 50 --max-jump 2000

仅依赖标准库，可直接在虚拟机运行。
"""
import csv
import math
import json
import argparse
from collections import defaultdict


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """两点大圆距离（米）"""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def shortest_angle_diff(a, b):
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def angle_lerp(a, b, t):
    return (a + shortest_angle_diff(a, b) * t) % 360.0


# 列名别名（大小写不敏感）
_ALIAS = {
    "lat": ["lat", "latitude", "纬度"],
    "lon": ["lon", "lng", "long", "longitude", "经度"],
    "sog": ["sog", "speed", "speedoverground", "航速", "速度"],
    "cog": ["cog", "course", "courseoverground", "heading", "航向", "航向角"],
    "ts": ["timestamp", "time", "ts", "utc", "时间戳", "时间"],
    "mmsi": ["mmsi", "shipid", "id", "编号", "船舶id"],
}


def detect_columns(header):
    """自动识别列名，返回 {标准名: 列下标}"""
    h = [str(c).strip().lower() for c in header]
    mapping = {}
    for std, names in _ALIAS.items():
        for i, c in enumerate(h):
            if c in names and std not in mapping:
                mapping[std] = i
                break
    return mapping


def read_ais(path):
    """读入 CSV，返回 (records, has_mmsi)，record 为 dict"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = detect_columns(header)
        assert "lat" in cols and "lon" in cols and "ts" in cols, \
            f"无法识别必要列 lat/lon/timestamp，实际表头: {header}"

        def g(row, key):
            i = cols.get(key)
            return float(row[i]) if (i is not None and row[i] not in ("", "-1")) else None

        records = []
        for row in reader:
            if not row or row[0].strip() == "":
                continue
            rec = {
                "lat": g(row, "lat"), "lon": g(row, "lon"),
                "sog": g(row, "sog"), "cog": g(row, "cog"),
                "ts": g(row, "ts"),
                "mmsi": int(float(row[cols["mmsi"]])) if ("mmsi" in cols and row[cols["mmsi"]] not in ("", "-1")) else None,
            }
            if rec["lat"] is None or rec["lon"] is None or rec["ts"] is None:
                continue
            records.append(rec)
    return records, ("mmsi" in cols)


def clean_track(records, max_speed, max_jump):
    """单条轨迹清洗，返回 (clean_records, stats)"""
    stats = {"total": len(records), "dup_removed": 0,
             "jump_removed": 0, "speed_removed": 0, "course_removed": 0}

    # 1. 排序
    records = sorted(records, key=lambda r: r["ts"])
    # 2. 去重（相同时间戳保留第一条）
    seen = set()
    dedup = []
    for r in records:
        key = round(r["ts"], 3)
        if key in seen:
            stats["dup_removed"] += 1
            continue
        seen.add(key)
        dedup.append(r)

    # 3. 异常检测
    clean = []
    prev = None
    for r in dedup:
        drop = False
        # 航速越界
        if r["sog"] is not None and not (0 <= r["sog"] <= max_speed):
            stats["speed_removed"] += 1
            drop = True
        # 航向越界
        if r["cog"] is not None and not (0 <= r["cog"] < 360):
            stats["course_removed"] += 1
            drop = True
        # 位置跳变（相邻点距离突变）
        if prev is not None and not drop:
            d = haversine(prev["lat"], prev["lon"], r["lat"], r["lon"])
            dt = max(r["ts"] - prev["ts"], 1e-6)
            if d > max_jump and d / dt > max_speed * 0.514 * 3:  # 超最大速度3倍且超距离阈值
                stats["jump_removed"] += 1
                drop = True
        if drop:
            continue
        clean.append(r)
        prev = r

    stats["kept"] = len(clean)
    return clean, stats


def interpolate_track(records, dt):
    """线性插值到等间隔时间网格（航向用最短角差插值）"""
    if len(records) < 2:
        return records
    out = []
    for i in range(len(records) - 1):
        a, b = records[i], records[i + 1]
        out.append(a)
        gap = b["ts"] - a["ts"]
        if gap > dt * 1.5:  # 有缺口，补点
            n = int(round(gap / dt)) - 1
            for k in range(1, n + 1):
                t = k / (n + 1)
                ts = a["ts"] + gap * t
                def lerp(x, y): return None if (x is None or y is None) else x + (y - x) * t
                rec = {
                    "lat": lerp(a["lat"], b["lat"]),
                    "lon": lerp(a["lon"], b["lon"]),
                    "sog": lerp(a["sog"], b["sog"]),
                    "cog": angle_lerp(a["cog"] or 0, b["cog"] or 0, t) if (a["cog"] is not None and b["cog"] is not None) else None,
                    "ts": ts,
                    "mmsi": a["mmsi"],
                }
                out.append(rec)
    out.append(records[-1])
    return out


def main():
    ap = argparse.ArgumentParser(description="AIS/ARPA 轨迹清洗")
    ap.add_argument("-i", "--input", required=True, help="输入 CSV 路径")
    ap.add_argument("-o", "--output", default="clean_data.csv", help="输出 CSV 路径")
    ap.add_argument("--max-speed", type=float, default=50.0, help="最大航速(节)，默认 50")
    ap.add_argument("--max-jump", type=float, default=2000.0, help="相邻点最大跳变距离(米)，默认 2000")
    ap.add_argument("--interpolate", type=float, default=None, help="插值到等间隔(秒)，如 5")
    ap.add_argument("--report", default="clean_report.json", help="清洗报告路径")
    args = ap.parse_args()

    records, has_mmsi = read_ais(args.input)
    print(f"读入 {len(records)} 条记录，{'含' if has_mmsi else '不含'} MMSI 列")

    # 按 MMSI 分组（无 MMSI 则单轨迹）
    groups = defaultdict(list)
    for r in records:
        groups[r["mmsi"]].append(r)

    all_clean, total_stats = [], defaultdict(int)
    for key, recs in groups.items():
        if args.interpolate:
            recs, _ = clean_track(recs, args.max_speed, args.max_jump)
            recs = interpolate_track(recs, args.interpolate)
            clean, stats = recs, {"kept": len(recs)}
        else:
            clean, stats = clean_track(recs, args.max_speed, args.max_jump)
        all_clean.extend(clean)
        for k, v in stats.items():
            total_stats[k] += v
        if key is not None:
            print(f"  MMSI {key}: 保留 {stats['kept']}/{len(recs)}")

    # 写输出（保持 lat/lon/cog/sog/timestamp 列序，含 mmsi 则加列）
    all_clean = sorted(all_clean, key=lambda r: r["ts"])
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = (["mmsi"] if has_mmsi else []) + ["lat", "lon", "cog", "sog", "timestamp"]
        w.writerow(header)
        for r in all_clean:
            row = ([r["mmsi"]] if has_mmsi else []) + [
                round(r["lat"], 6), round(r["lon"], 6),
                round(r["cog"], 2) if r["cog"] is not None else "",
                round(r["sog"], 2) if r["sog"] is not None else "",
                round(r["ts"], 3)]
            w.writerow(row)

    # 写报告
    total_stats["total"] = len(records)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(total_stats, f, ensure_ascii=False, indent=2)

    print(f"\n清洗完成: {total_stats.get('total', 0)} -> {total_stats.get('kept', len(all_clean))} 条")
    print(f"  去重 {total_stats.get('dup_removed', 0)} / 位置跳变 {total_stats.get('jump_removed', 0)} / "
          f"航速异常 {total_stats.get('speed_removed', 0)} / 航向异常 {total_stats.get('course_removed', 0)}")
    print(f"输出: {args.output}  报告: {args.report}")


if __name__ == "__main__":
    main()
