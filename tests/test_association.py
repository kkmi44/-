#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_association.py —— 第四章关联算法单元测试

运行: python tests/test_association.py
验证论文 4.1 节的四个核心模块：运动学关联、各向异性高斯、一致性校验、级联关联。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ch4.association import (
    kinematic_confidence, spatial_confidence,
    consistency_check, cascade_associate,
)


def test_kinematic():
    """运动学关联：噪声轨迹应恢复正确对应"""
    rng = np.random.default_rng(0)
    T = 10
    ais = np.array([
        np.stack([np.linspace(0, 50, T), np.linspace(0, 10, T)], axis=1),
        np.stack([np.linspace(100, 50, T), np.linspace(0, 20, T)], axis=1),
        np.stack([np.linspace(0, 0, T), np.linspace(100, 0, T)], axis=1),
    ])
    radar = np.array([ais[2] + rng.normal(0, 2, (T, 2)),
                      ais[0] + rng.normal(0, 2, (T, 2)),
                      ais[1] + rng.normal(0, 2, (T, 2))])
    _, matches = kinematic_confidence(radar, ais)
    assert matches == [(0, 2), (1, 0), (2, 1)], f"匹配错误: {matches}"
    print("✓ 运动学关联：正确恢复 0->2, 1->0, 2->1")


def test_spatial():
    """各向异性高斯：纵向衰减应快于横向"""
    boxes = [(300, 200, 80, 20)]   # 扁框，宽80高20
    pts = [(300, 200), (300, 250)]  # 中心 vs 纵向偏移50
    Conf = spatial_confidence(pts, boxes)
    assert abs(Conf[0, 0] - 1.0) < 1e-6, "中心点置信度应为1"
    assert Conf[1, 0] < 0.01, "纵向偏移50应几乎衰减到0"
    print("✓ 各向异性高斯：纵向衰减远快于横向")


def test_consistency():
    """一致性校验：闭环/冲突/降级"""
    assert consistency_check(0.9, 0.85, 0.8, 0.5)[0] == "RAV"
    assert consistency_check(0.9, 0.6, 0.1, 0.5)[0] == "RA"   # 剪视觉
    assert consistency_check(0.6, 0.9, 0.1, 0.5)[0] == "RV"   # 剪AIS
    assert consistency_check(0.9, None, None)[0] == "RA"
    assert consistency_check(None, 0.9, None)[0] == "RV"
    assert consistency_check(None, None, None)[0] == "R"
    print("✓ 一致性校验：闭环/冲突消解/降级全部正确")


def test_cascade_rav():
    """三源一致时应闭环输出 RAV"""
    rng = np.random.default_rng(1)
    T = 8
    # 两条船，三源坐标一致
    ais = np.array([
        np.stack([np.linspace(0, 40, T), np.linspace(0, 0, T)], axis=1),
        np.stack([np.linspace(200, 160, T), np.linspace(0, 10, T)], axis=1),
    ])
    radar = np.array([ais[0] + rng.normal(0, 1, (T, 2)),
                      ais[1] + rng.normal(0, 1, (T, 2))])
    boxes = [(100, 100, 40, 20), (400, 120, 50, 25)]
    radar_px = np.array([[100, 100], [400, 120]])   # 雷达投影和框一致
    ais_px = np.array([[100, 100], [400, 120]])     # AIS 投影和框一致
    fused = cascade_associate(["R1", "R2"], radar, boxes,
                              [413000001, 413000002], ais, radar_px, ais_px)
    assert all(f["source"] == "RAV" for f in fused), f"应全部闭环: {fused}"
    assert fused[0]["mmsi"] == 413000001
    print(f"✓ 级联关联：三源一致时闭环输出 {[f['source'] for f in fused]}")


if __name__ == "__main__":
    test_kinematic()
    test_spatial()
    test_consistency()
    test_cascade_rav()
    print("\n全部测试通过 ✓")
