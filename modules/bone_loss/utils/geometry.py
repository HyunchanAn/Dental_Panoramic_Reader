"""
Geometry and Calculation Utilities for Pano_BoneLoss_Measurement.

이 모듈은 방사선학적 랜드마크 사이의 거리 연산, 골소실(RBL) 퍼센티지 산출,
그리고 픽셀-물리단위 변환 등의 수학적 연산을 제공합니다.
"""

import math
from typing import Tuple

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    두 2D 좌표 간의 유클리드 거리를 계산합니다.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_rbl(cej: Tuple[float, float], crest: Tuple[float, float], apex: Tuple[float, float]) -> float:
    """
    치은-백악질 경계(CEJ), 치조골정(Crest), 치근단(Apex) 좌표를 기반으로 
    방사선학적 골소실(RBL, Radiographic Bone Loss) 퍼센티지를 계산합니다.
    
    CEJ에서 Apex까지의 거리를 전체 치근 길이로, CEJ에서 Crest까지의 거리를 골소실 길이로 계산합니다.
    만약 Crest가 CEJ보다 치관 측(coronal)에 위치할 경우 (음수 산출 시) 0.0%로 Clamping 처리합니다.
    """
    root_length = calculate_distance(cej, apex)
    if root_length == 0:
        raise ValueError("CEJ와 Apex가 동일한 위치에 있을 수 없습니다 (Root length is 0).")
        
    bone_loss_distance = calculate_distance(cej, crest)
    
    # CEJ-Crest 벡터와 CEJ-Apex 벡터의 내적을 확인하여 같은 방향인지 점검.
    # 내적이 0 미만이면 Crest가 CEJ 기준 치관 방향에 있으므로 0% (정상) 처리
    vector_loss = (crest[0] - cej[0], crest[1] - cej[1])
    vector_root = (apex[0] - cej[0], apex[1] - cej[1])
    
    dot_product = vector_loss[0] * vector_root[0] + vector_loss[1] * vector_root[1]
    
    if dot_product < 0:
        return 0.0
        
    rbl_percentage = (bone_loss_distance / root_length) * 100.0
    return min(100.0, rbl_percentage)


def pixel_to_mm(pixel_distance: float, pixels_per_mm: float) -> float:
    """
    픽셀 거리를 물리적 밀리미터(mm) 단위로 변환합니다.
    """
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm은 0보다 커야 합니다.")
    return pixel_distance / pixels_per_mm
