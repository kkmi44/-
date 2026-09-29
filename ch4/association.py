#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ch4/association.py —— 第四章 多源数据关联匹配核心

实现论文 4.1 节的置信度级联关联（纯规则算法，无需训练）：
  1. 雷达-AIS 运动学关联（算法 4.1：时域滑窗平均欧氏距离 -> exp(-C/β) -> 匈牙利）
  2. 雷达-视觉 空间配准（算法 4.2：长宽解耦的各向异性高斯）
  3. AIS-视觉 辅助匹配（同各向异性高斯）
  4. 三源闭环一致性校验 + 冲突消解 + 残余关联（论文 4.1.2）

默认参数对齐论文 4.3.1：β=50, γ_penalty=0.75。

坐标约定：调用方负责把三源统一到可比较的空间——
  - 运动学关联：轨迹位置序列（米制或经纬度均可，两源需一致）
  - 空间关联：投影点到像素平面的点 + 检测框(中心+宽高)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# 论文 4.3.1 默认参数
BETA = 50.0            # 运动学置信度尺度平滑系数
ALPHA = 2.0            # 空间置信度尺度系数（σ=size/α）
GAMMA_PENALTY = 0.75   # 残余关联降级惩罚系数


# --------------------------------------------------------------------------
# 1. 雷达-AIS 运动学关联（算法 4.1）
# --------------------------------------------------------------------------
def trajectory_cost(rt, at):
    """单对轨迹的时域窗口平均欧氏距离 C[i,j] = mean_t ||rt(t) - at(t)||2"""
    rt = np.asarray(rt, dtype=float)
    at = np.asarray(at, dtype=float)
    d = np.linalg.norm(rt - at, axis=-1)      # 每时间步距离
    return float(np.mean(d))


def kinematic_confidence(radar_pos, ais_pos, beta=BETA):
    """雷达-AIS 运动学置信度矩阵 + 匈牙利最优匹配

    参数:
      radar_pos: (M, T, 2) 雷达轨迹位置序列
      ais_pos:   (N, T, 2) AIS 轨迹位置序列（已插值对齐到同一时间网格）
    返回:
      Conf: (M, N) 置信度矩阵 Conf[i,j] = exp(-C[i,j]/β)
      matches: [(i, j)] 匈牙利全局最优匹配对
    """
    M = len(radar_pos)
    N = len(ais_pos)
    Conf = np.zeros((M, N))
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            C[i, j] = trajectory_cost(radar_pos[i], ais_pos[j])
            Conf[i, j] = np.exp(-C[i, j] / beta)
    matches = _hungarian(C)
    return Conf, matches


# --------------------------------------------------------------------------
# 2/3. 空间关联：各向异性高斯（算法 4.2，雷达-视觉 与 AIS-视觉 共用）
# --------------------------------------------------------------------------
def spatial_confidence(points_px, boxes, alpha=ALPHA):
    """长宽解耦的各向异性高斯空间置信度

    参数:
      points_px: (P, 2) 投影点 (u, v)
      boxes:     (V, 4) 检测框 (uc, vc, w, h)（中心 + 宽高，像素）
    返回:
      ConfS: (P, V)  ConfS[p,v] = exp(-[(u-uc)²/2σu² + (v-vc)²/2σv²])，σu=w/α, σv=h/α
    """
    P = len(points_px)
    V = len(boxes)
    ConfS = np.zeros((P, V))
    points_px = np.asarray(points_px, dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    for v in range(V):
        uc, vc, w, h = boxes[v]
        su, sv = w / alpha, h / alpha        # 尺度自适应、长宽解耦
        du = points_px[:, 0] - uc
        dv = points_px[:, 1] - vc
        ConfS[:, v] = np.exp(-(du ** 2 / (2 * su ** 2) + dv ** 2 / (2 * sv ** 2)))
    return ConfS


def ais_vision_confidence(ais_points_px, boxes, alpha=ALPHA):
    """AIS-视觉 直接关联置信度（公式 4-5），复用各向异性高斯"""
    return spatial_confidence(ais_points_px, boxes, alpha)


# --------------------------------------------------------------------------
# 4. 三源闭环一致性校验 + 冲突消解（论文 4.1.2）
# --------------------------------------------------------------------------
def consistency_check(conf_ra, conf_rv, conf_av, thresh=0.5):
    """对单个雷达目标做三源闭环校验

    参数:
      conf_ra: 该雷达目标与其 AIS 匹配的置信度（None 表示未关联）
      conf_rv: 该雷达目标与其视觉匹配的置信度（None 表示未关联）
      conf_av: 对应的 AIS 与视觉之间的直接置信度（None 表示无法计算）
      thresh: 一致性门限
    返回:
      source: 'RAV' 三源全要素 / 'RA' 雷达-AIS / 'RV' 雷达-视觉 / 'R' 仅雷达
      conf:   综合置信度
    """
    has_ais = conf_ra is not None
    has_vis = conf_rv is not None

    # 只关联上一种 -> 直接二源输出
    if has_ais and not has_vis:
        return "RA", conf_ra
    if has_vis and not has_ais:
        return "RV", conf_rv
    if not has_ais and not has_vis:
        return "R", 0.0

    # 同时关联 AIS + 视觉 -> 三源闭环校验
    if conf_av is not None and conf_av >= thresh:
        # 闭环成功，输出全要素
        return "RAV", max(conf_ra, conf_rv, conf_av)

    # 校验失败（冲突）：按置信度高低剪一路
    if conf_ra >= conf_rv:
        return "RA", conf_ra          # 剪视觉，保雷达-AIS
    else:
        return "RV", conf_rv          # 剪 AIS，保雷达-视觉（匿名实体）


# --------------------------------------------------------------------------
# 残余关联（论文 4.1.2）
# --------------------------------------------------------------------------
def residual_associate(ais_points_px, vision_boxes, alpha=ALPHA, gamma=GAMMA_PENALTY):
    """残余池中 AIS 与视觉直接配对，输出降级二源目标

    返回: 匹配对 [(i, j)] 及综合置信度 Conf_state = γ · Conf_AV
    """
    ConfAV = ais_vision_confidence(ais_points_px, vision_boxes, alpha)
    matches = _hungarian(1.0 - ConfAV)   # 置信度转成本，越大越匹配
    conf_states = [gamma * ConfAV[i, j] for i, j in matches]
    return matches, conf_states


# --------------------------------------------------------------------------
# 完整级联关联（主入口）
# --------------------------------------------------------------------------
def cascade_associate(radar_ids, radar_pos, vision_boxes, ais_mmsi, ais_pos,
                      radar_px, ais_px,
                      thresh=0.5, beta=BETA, alpha=ALPHA, gamma=GAMMA_PENALTY):
    """多维置信度级联关联（论文 4.1 完整流程）

    参数:
      radar_ids:    [id] 雷达目标 ID 列表（M 个）
      radar_pos:    (M, T, 2) 雷达轨迹位置序列（米制，用于雷达-AIS 运动学关联）
      vision_boxes: [(uc, vc, w, h)] 视觉检测框（V 个）
      ais_mmsi:     [mmsi] AIS 目标 MMSI 列表（N 个）
      ais_pos:      (N, T, 2) AIS 轨迹位置序列（米制，已对齐到同一时间网格）
      radar_px:     (M, 2) 雷达目标投影到视觉像素平面（用于雷达-视觉空间关联）
      ais_px:       (N, 2) AIS 目标投影到视觉像素平面（用于 AIS-视觉空间关联）
      thresh: 一致性门限
    返回:
      fused: [{target_id, mmsi, source, conf}] 融合目标列表
    """
    fused = []

    # 1. 雷达-AIS 运动学关联
    ConfRA, ra_matches = kinematic_confidence(radar_pos, ais_pos, beta)
    # 2. 雷达-视觉 空间关联
    ConfRV = spatial_confidence(radar_px, vision_boxes, alpha)
    # 3. AIS-视觉 空间关联
    ConfAV = ais_vision_confidence(ais_px, vision_boxes, alpha)

    ra_map = {i: j for i, j in ra_matches}
    rv_map = _hungarian_map(1.0 - ConfRV)

    # 以雷达为核心做一致性校验
    for i, rid in enumerate(radar_ids):
        j = ra_map.get(i)
        k = rv_map.get(i)
        conf_ra = float(ConfRA[i, j]) if j is not None else None
        conf_rv = float(ConfRV[i, k]) if k is not None else None
        conf_av = float(ConfAV[j, k]) if (j is not None and k is not None) else None

        source, conf = consistency_check(conf_ra, conf_rv, conf_av, thresh)
        mmsi = ais_mmsi[j] if (j is not None and source in ("RAV", "RA")) else None
        fused.append({"target_id": rid, "mmsi": mmsi, "source": source, "conf": round(conf, 4)})

    return fused


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _hungarian(cost):
    """匈牙利全局最优匹配，返回 [(i, j)]（最小化 cost）"""
    if cost is None or cost.size == 0:
        return []
    row, col = linear_sum_assignment(cost)
    return list(zip(row.tolist(), col.tolist()))


def _hungarian_map(cost):
    return {i: j for i, j in _hungarian(cost)}
