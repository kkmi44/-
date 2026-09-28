# -*- coding: utf-8 -*-
"""
common/types.py —— 统一数据模型（全工程共用）

三章之间的数据都通过这里定义的标准结构传递，避免靠猜字段名对接。
使用 dataclass 保证类型清晰、可序列化。
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any


@dataclass
class BBox:
    """像素边界框 (xmin, ymin, xmax, ymax)"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def wh(self) -> Tuple[float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin)


@dataclass
class VisionBox:
    """视觉检测框（第二章输出 → 第四章视觉输入）
    中心 (u, v) + 宽高 (w, h)，单位像素"""
    timestamp: float
    cls: str = "ship"
    conf: float = 1.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    h: float = 0.0
    track_id: Optional[int] = None

    def to_bbox(self) -> BBox:
        return BBox(self.u - self.w / 2, self.v - self.h / 2,
                    self.u + self.w / 2, self.v + self.h / 2)


@dataclass
class RadarPoint:
    """雷达点迹（第四章雷达信源）"""
    timestamp: float
    rng: float = 0.0        # 距离(米)
    brg: float = 0.0        # 方位(度)
    lat: float = 0.0        # 投影后大地坐标
    lon: float = 0.0
    track_id: Optional[int] = None


@dataclass
class AISReport:
    """AIS 报文（第三/四章）
    字段与 ARPA 报文高度重叠，可互相映射"""
    timestamp: float
    mmsi: int
    lat: float
    lon: float
    sog: float = 0.0        # 对地航速(节)
    cog: float = 0.0        # 对地航向(度)
    name: str = ""
    ship_type: str = ""


@dataclass
class RadarTarget:
    """第三章雷达目标（预处理产物）"""
    frame_id: int
    target_id: int                     # 跨帧关联后的全局船舶ID
    centroid: Tuple[float, float]      # (x, y) 像素质心
    bbox: BBox
    patch: Any = None                  # 224×224 灰度图 (0~1)


@dataclass
class FusedTarget:
    """第四章综合感知目标（最终输出）"""
    target_id: str
    mmsi: Optional[str] = None         # 无 AIS 时为 None（非合作目标）
    bbox: Optional[BBox] = None        # 无视觉时为 None
    position: Optional[Tuple[float, float]] = None   # (lat, lon)
    sog: float = 0.0
    cog: float = 0.0
    conf: float = 0.0                  # 综合置信度
    source: str = "R"                  # "RAV"/"RA"/"RV"/"AV"/"R"
