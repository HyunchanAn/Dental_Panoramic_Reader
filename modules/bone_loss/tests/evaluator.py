import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import mean_absolute_error, r2_score

class Evaluator:
    """
    실제 임상의가 측정한 Ground Truth와 AI 추론 결과를 비교하여
    MAE 및 R-squared 점수를 산출하는 검증 모듈입니다.
    """
    
    def __init__(self):
        self.gt_bone_losses = []
        self.pred_bone_losses = []
        
        self.gt_coords = []
        self.pred_coords = []

    def add_record(self, 
                   gt_rbl: float, pred_rbl: float, 
                   gt_landmarks: Dict[str, Tuple[float, float]] = None, 
                   pred_landmarks: Dict[str, Tuple[float, float]] = None):
        """
        개별 치아에 대한 평가 기록을 추가합니다.
        """
        self.gt_bone_losses.append(gt_rbl)
        self.pred_bone_losses.append(pred_rbl)
        
        if gt_landmarks and pred_landmarks:
            for key in ["mesial_cej", "distal_cej", "mesial_crest", "distal_crest", "root_apex"]:
                if key in gt_landmarks and key in pred_landmarks:
                    self.gt_coords.extend(list(gt_landmarks[key]))
                    self.pred_coords.extend(list(pred_landmarks[key]))

    def calculate_rbl_r2_score(self) -> float:
        """
        골흡수율 분율(RBL %)에 대한 R-squared 스코어를 산출합니다.
        """
        if len(self.gt_bone_losses) < 2:
            return 0.0
        return r2_score(self.gt_bone_losses, self.pred_bone_losses)
        
    def calculate_rbl_mae(self) -> float:
        """
        골흡수율 분율(RBL %)에 대한 MAE를 산출합니다.
        """
        if not self.gt_bone_losses:
            return 0.0
        return mean_absolute_error(self.gt_bone_losses, self.pred_bone_losses)

    def calculate_coordinate_mae(self) -> float:
        """
        예측된 랜드마크 좌표계와 Ground Truth 좌표계 간의 MAE를 산출합니다.
        """
        if not self.gt_coords:
            return 0.0
        return mean_absolute_error(self.gt_coords, self.pred_coords)
