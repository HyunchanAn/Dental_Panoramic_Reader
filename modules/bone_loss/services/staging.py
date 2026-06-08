"""
Periodontitis Staging Logic Module.

2017 World Workshop 기준을 바탕으로 환자의 치주염 Stage(I~IV)와 Extent(국소적/전반적)를 분류합니다.
"""

from typing import List, Dict, Tuple
from enum import Enum

class Stage(str, Enum):
    NORMAL = "Normal"
    STAGE_I = "Stage I"
    STAGE_II = "Stage II"
    STAGE_III = "Stage III"
    STAGE_IV = "Stage IV"

class Extent(str, Enum):
    LOCALIZED = "Localized"
    GENERALIZED = "Generalized"

def determine_site_stage(rbl: float) -> Stage:
    """개별 측정 부위의 골소실 퍼센티지(RBL)에 따른 초기 병기를 산출합니다."""
    if rbl < 15.0:
        if rbl > 0.0:
            return Stage.STAGE_I
        return Stage.NORMAL
    elif 15.0 <= rbl < 33.0:
        return Stage.STAGE_II
    else: 
        return Stage.STAGE_III # Stage IV 판단은 Complexity 요인이 필요함

def determine_patient_stage(
    tooth_metrics: List[Dict[str, float]], 
    has_severe_complexity: bool = False,
    teeth_lost_due_to_perio: int = 0
) -> Tuple[Stage, Extent]:
    """
    전체 치아의 측정 데이터를 취합하여 환자 단위의 치주염 Stage 및 Extent를 산출합니다.
    
    Args:
        tooth_metrics: 각 치아별 최대 RBL(%) 정보. 예: [{"tooth": 11, "max_rbl": 20.5}, ...]
        has_severe_complexity: 교합 붕괴 등 중증 복잡도 요소 존재 여부.
        teeth_lost_due_to_perio: 치주염으로 상실된 치아 수.
        
    Returns:
        (최종 Stage, Extent) 튜플을 반환.
    """
    if not tooth_metrics:
        return Stage.NORMAL, Extent.LOCALIZED

    max_rbl_overall = max([metric.get("max_rbl", 0.0) for metric in tooth_metrics])
    
    # 1. 초기 Stage 결정
    initial_stage = determine_site_stage(max_rbl_overall)
    
    # 2. Stage III / IV 판별 (복잡도 고려)
    final_stage = initial_stage
    if max_rbl_overall >= 33.0:
        if has_severe_complexity or teeth_lost_due_to_perio >= 5:
            final_stage = Stage.STAGE_IV
        else:
            final_stage = Stage.STAGE_III
            
    # 3. Extent 판별 (> 30% 치아가 RBL 15% 이상인지 여부)
    affected_threshold = 15.0 
    affected_teeth_count = sum(1 for metric in tooth_metrics if metric.get("max_rbl", 0.0) >= affected_threshold)
    total_teeth = len(tooth_metrics)
    
    extent = Extent.LOCALIZED
    if (affected_teeth_count / total_teeth) * 100 > 30.0:
        extent = Extent.GENERALIZED
        
    return final_stage, extent
