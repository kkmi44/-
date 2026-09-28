#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_synthetic.py —— 合成多源船舶数据集生成器

生成"雷达回波帧 + ARPA/AIS 报文 + 视觉检测框 + 真值"的合成数据，
用于三章算法的离线测试、单元测试与演示系统。真实数据到位后替换即可。

运行：
    python make_synthetic.py --scenes 3 --frames 120 --ships 4 --out data/synthetic

仅依赖 numpy + 标准库。
"""
import os
import csv
import argparse
import numpy as np


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
class Config:
    # 时空
    period = 2.0                 # 雷达帧周期(秒)
    world_range = 2000.0         # 雷达探测范围(米)，单边
    radar_size = 256             # 雷达图边长(像素)
    vis_size = 640               # 视觉图边长(像素)
    # 噪声
    clutter_level = 0.03         # 海杂波基础噪声强度
    clutter_spark = 6            # 每帧随机亮点个数(模拟瞬时杂波)
    # 相机(简化针孔，朝向本船正前)
    fx = 800.0
    fy = 800.0
    cx = 320.0
    cy = 240.0
    cam_height = 15.0            # 相机高度(米)

    # 场景规模
    frames = 120                 # 帧数
    ships = 4                    # 船舶数
    # 随机种子
    seed = 42


# --------------------------------------------------------------------------
# 船舶轨迹生成(真实值)：匀速 + 轻微转向机动
# --------------------------------------------------------------------------
def gen_ship_tracks(cfg: Config, rng: np.random.Generator):
    tracks = []
    n = cfg.frames
    for i in range(cfg.ships):
        # 初始位置：分布在四周(米)，本船在原点
        angle0 = rng.uniform(0, 2 * np.pi)
        dist0 = rng.uniform(300, cfg.world_range * 0.8)
        x0 = dist0 * np.cos(angle0)
        y0 = dist0 * np.sin(angle0)
        # 初始航向(度)与航速(米/秒 -> 节)
        cog = rng.uniform(0, 360)
        sog_ms = rng.uniform(3, 12)          # 米/秒
        sog_kn = sog_ms * 1.94384            # 节
        # 机动参数
        turn_rate = rng.uniform(-2.0, 2.0)   # 度/步

        mmsi = 100000000 + i * 111111

        xs, ys, cogs, sogs = [], [], [], []
        x, y, c = x0, y0, cog
        for t in range(n):
            xs.append(x); ys.append(y)
            cogs.append(c % 360)
            sogs.append(sog_kn)
            c += turn_rate
            x += sog_ms * cfg.period * np.sin(np.radians(c))
            y += sog_ms * cfg.period * np.cos(np.radians(c))

        tracks.append({
            "id": i,
            "mmsi": mmsi,
            "x": np.array(xs), "y": np.array(ys),
            "cog": np.array(cogs), "sog_kn": np.array(sogs),
        })
    return tracks


# --------------------------------------------------------------------------
# 雷达帧渲染：高斯亮斑(船舶回波) + 海杂波
# --------------------------------------------------------------------------
def render_radar_frame(cfg: Config, tracks, t: int, rng: np.random.Generator):
    S = cfg.radar_size
    img = np.full((S, S), cfg.clutter_level, dtype=np.float32)
    c = S / 2.0
    scale = S / (2 * cfg.world_range)          # 像素/米

    for tr in tracks:
        px = c + tr["x"][t] * scale
        py = c - tr["y"][t] * scale            # 北向上 -> 图像向下
        if not (0 <= px < S and 0 <= py < S):
            continue
        # 回波强度随距离衰减
        dist = np.hypot(tr["x"][t], tr["y"][t])
        intensity = 1.0 * np.exp(-dist / cfg.world_range)
        # 高斯亮斑
        xx, yy = np.meshgrid(np.arange(S), np.arange(S))
        g = intensity * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 3.0 ** 2))
        img += g

    # 海杂波：低强度均匀噪声 + 瞬时随机亮点
    img += rng.uniform(0, cfg.clutter_level, size=(S, S))
    for _ in range(cfg.clutter_spark):
        sx, sy = rng.integers(0, S, 2)
        img[sy, sx] += rng.uniform(0.3, 0.8)

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# ARPA/AIS 报文生成：非均匀采样(模拟真实播报) + 测量噪声
# --------------------------------------------------------------------------
def gen_messages(cfg: Config, tracks, rng: np.random.Generator, kind="arpa"):
    """生成非均匀采样的报文，字段: timestamp,mmsi,lat,lon,sog,cog
    kind='arpa' 采样较密(~2s 但随机缺失)，kind='ais' 采样较疏(30s~3min)"""
    rows = []
    base_lat, base_lon = 30.563549, 114.306087   # 武汉长江附近基准点
    for tr in tracks:
        if kind == "arpa":
            # 约每 2~4 秒一条，随机缺失模拟遮挡
            for t in range(cfg.frames):
                if rng.random() < 0.3:            # 30% 缺失
                    continue
                ts = t * cfg.period + rng.uniform(0, 0.5)
                rows.append(_msg(tr, t, ts, base_lat, base_lon, rng, noise=1.0))
        else:  # ais
            t = 0
            while t < cfg.frames:
                ts = t * cfg.period
                rows.append(_msg(tr, t, ts, base_lat, base_lon, rng, noise=3.0))
                t += int(rng.uniform(15, 90) / cfg.period)   # 30s~180s
    rows.sort(key=lambda r: r[0])
    return rows


def _msg(tr, t, ts, base_lat, base_lon, rng, noise):
    # 米 -> 经纬度近似(纬度方向 110540 m/deg, 经度方向随纬度收缩)
    lat = base_lat + tr["y"][t] / 110540.0
    lon = base_lon + tr["x"][t] / (111320.0 * np.cos(np.radians(base_lat)))
    sog = tr["sog_kn"][t] + rng.normal(0, 0.1)
    cog = (tr["cog"][t] + rng.normal(0, noise)) % 360
    return [round(ts, 2), tr["mmsi"], round(lat, 6), round(lon, 6),
            round(sog, 2), round(cog, 1)]


# --------------------------------------------------------------------------
# 视觉框生成：世界坐标 -> 像素(简化针孔 + 高度补偿)
# --------------------------------------------------------------------------
def gen_vision_boxes(cfg: Config, tracks, t: int, rng: np.random.Generator):
    """把船舶相对位置投影到像素，返回 [timestamp, cls, conf, u, v, w, h]"""
    boxes = []
    for tr in tracks:
        # 相机坐标系：x向右，y向下，z向前(本船前方=北)
        X = tr["x"][t]            # 东 -> 相机右
        Z = tr["y"][t]            # 北 -> 相机前
        Y = -cfg.cam_height       # 世界高 -> 相机下
        if Z <= 5:                # 在身后或过近，跳过
            continue
        u = cfg.fx * X / Z + cfg.cx
        v = cfg.fy * Y / Z + cfg.cy
        # 目标尺度：距离越远框越小
        s = max(4.0, 4000.0 / Z)
        w = h = s * (0.8 + 0.4 * rng.random())
        if not (0 <= u < cfg.vis_size and 0 <= v < cfg.vis_size):
            continue
        boxes.append([round(t * cfg.period, 2), "ship", round(rng.uniform(0.7, 0.99), 3),
                      round(u, 1), round(v, 1), round(w, 1), round(h, 1)])
    return boxes


# --------------------------------------------------------------------------
# 真值
# --------------------------------------------------------------------------
def gen_gt(cfg: Config, tracks):
    rows = []
    base_lat, base_lon = 30.563549, 114.306087
    for t in range(cfg.frames):
        for tr in tracks:
            lat = base_lat + tr["y"][t] / 110540.0
            lon = base_lon + tr["x"][t] / (111320.0 * np.cos(np.radians(base_lat)))
            rows.append([t, tr["id"], tr["mmsi"], round(lat, 6), round(lon, 6),
                         round(tr["sog_kn"][t], 2), round(tr["cog"][t], 1)])
    return rows


# --------------------------------------------------------------------------
# 保存与主流程
# --------------------------------------------------------------------------
def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_scene(cfg: Config, scene_idx: int, out_root: str, rng: np.random.Generator):
    scene_dir = os.path.join(out_root, f"scene_{scene_idx:03d}")
    radar_dir = os.path.join(scene_dir, "radar")
    os.makedirs(radar_dir, exist_ok=True)

    tracks = gen_ship_tracks(cfg, rng)
    print(f"  scene_{scene_idx:03d}: {cfg.ships} 艘船, {cfg.frames} 帧")

    # 雷达帧
    for t in range(cfg.frames):
        img = render_radar_frame(cfg, tracks, t, rng)
        np.save(os.path.join(radar_dir, f"{t:05d}.npy"), img)

    # 报文
    arpa = gen_messages(cfg, tracks, rng, "arpa")
    save_csv(os.path.join(scene_dir, "arpa.csv"),
             ["timestamp", "mmsi", "lat", "lon", "sog", "cog"], arpa)
    ais = gen_messages(cfg, tracks, rng, "ais")
    save_csv(os.path.join(scene_dir, "ais.csv"),
             ["timestamp", "mmsi", "lat", "lon", "sog", "cog"], ais)

    # 视觉框(逐帧)
    vis_rows = []
    for t in range(cfg.frames):
        vis_rows += gen_vision_boxes(cfg, tracks, t, rng)
    save_csv(os.path.join(scene_dir, "vision.csv"),
             ["timestamp", "cls", "conf", "u", "v", "w", "h"], vis_rows)

    # 真值
    save_csv(os.path.join(scene_dir, "gt.csv"),
             ["frame", "id", "mmsi", "lat", "lon", "sog", "cog"], gen_gt(cfg, tracks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ships", type=int, default=4)
    ap.add_argument("--out", type=str, default="data/synthetic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config()
    cfg.frames = args.frames
    cfg.ships = args.ships
    cfg.seed = args.seed

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"生成合成数据集: {args.scenes} 场景 -> {args.out}")
    for s in range(args.scenes):
        make_scene(cfg, s, args.out, rng)

    # 写说明
    with open(os.path.join(args.out, "README.txt"), "w", encoding="utf-8") as f:
        f.write("合成数据集说明\n")
        f.write(f"场景数={args.scenes} 帧数={args.frames} 船舶数={args.ships} 周期={cfg.period}s\n")
        f.write("目录结构:\n")
        f.write("  radar/      雷达回波帧 *.npy (灰度 0~255, 本船在中心)\n")
        f.write("  arpa.csv    ARPA报文 [timestamp,mmsi,lat,lon,sog,cog] 非均匀采样\n")
        f.write("  ais.csv     AIS报文 [timestamp,mmsi,lat,lon,sog,cog] 稀疏采样\n")
        f.write("  vision.csv  视觉框 [timestamp,cls,conf,u,v,w,h]\n")
        f.write("  gt.csv      真值 [frame,id,mmsi,lat,lon,sog,cog]\n")
        f.write("基准点经纬度: 30.563549, 114.306087 (武汉长江段)\n")

    print("完成。目录结构见 README.txt")


if __name__ == "__main__":
    main()
