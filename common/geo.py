# -*- coding: utf-8 -*-
"""
common/geo.py —— 坐标转换（全工程共用）

三套坐标系：
  1. 大地坐标系 WGS-84（经纬度，度）—— AIS/ARPA 报文
  2. 局部平面坐标系（米）—— 雷达测距测向、轨迹匹配、距离计算
  3. 像素坐标系（像素）—— 视觉检测框、雷达图像

所有坐标转换集中在此，各模块禁止私自定义，避免量纲错乱。
"""
import math
import numpy as np

# 地球近似常数（等距圆柱投影）
_M_PER_DEG_LAT = 110540.0          # 纬度方向 米/度
_M_PER_DEG_LON_BASE = 111320.0     # 经度方向 米/度（赤道处）


# --------------------------------------------------------------------------
# 经纬度 <-> 局部米制（等距圆柱近似，适合小范围港口/航道）
# --------------------------------------------------------------------------
def ll2xy(lat, lon, ref_lat, ref_lon):
    """经纬度 -> 局部米制 (x=东, y=北)，以 (ref_lat, ref_lon) 为原点"""
    x = (lon - ref_lon) * _M_PER_DEG_LON_BASE * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * _M_PER_DEG_LAT
    return float(x), float(y)


def xy2ll(x, y, ref_lat, ref_lon):
    """局部米制 -> 经纬度"""
    lat = ref_lat + y / _M_PER_DEG_LAT
    lon = ref_lon + x / (_M_PER_DEG_LON_BASE * math.cos(math.radians(ref_lat)))
    return float(lat), float(lon)


def haversine(lat1, lon1, lat2, lon2):
    """两点间大圆距离（米），用于真实距离计算/异常检测"""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# 世界 <-> 相机 <-> 像素（针孔模型）
# --------------------------------------------------------------------------
def world2camera(Pw, R, T):
    """世界坐标(3,) -> 相机坐标，Pc = R·Pw + T"""
    Pw = np.asarray(Pw, dtype=float)
    return np.asarray(R, dtype=float) @ Pw + np.asarray(T, dtype=float)


def camera2pixel(Pc, K):
    """相机坐标(3,) -> 像素 (u, v)，针孔模型 u = fx*X/Z + cx"""
    Pc = np.asarray(Pc, dtype=float)
    K = np.asarray(K, dtype=float)
    if Pc[2] <= 0:
        return None
    u = K[0, 0] * Pc[0] / Pc[2] + K[0, 2]
    v = K[1, 1] * Pc[1] / Pc[2] + K[1, 2]
    return float(u), float(v)


def ll2pixel(lat, lon, ref_lat, ref_lon, R, T, K):
    """经纬度 -> 像素（第四章雷达/视觉、AIS/视觉投影配准用）"""
    x, y = ll2xy(lat, lon, ref_lat, ref_lon)
    Pw = np.array([x, y, 0.0])        # 海平面高度 0
    Pc = world2camera(Pw, R, T)
    return camera2pixel(Pc, K)


# --------------------------------------------------------------------------
# 角度工具
# --------------------------------------------------------------------------
def norm_angle(deg):
    """归一化到 [0, 360)"""
    return deg % 360.0


def shortest_angle_diff(a, b):
    """最短角差（度），返回 [-180, 180)，用于航向插值/方位差"""
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def angle_lerp(a, b, t):
    """航向等角度量的线性插值（避免 360° 跳变）"""
    diff = shortest_angle_diff(a, b)
    return norm_angle(a + diff * t)
