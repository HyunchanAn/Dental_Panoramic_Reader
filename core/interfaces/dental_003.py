import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../../Dental_003"))
if module_path not in sys.path:
    sys.path.append(module_path)

def init_003_model():
    """Dental_003 (치조골 랜드마크 SAM) 모델을 초기화합니다. 미구현 상태이므로 None 반환."""
    return None

def calculate_bone_loss(image: np.ndarray, tooth_roi_data: dict, model) -> dict:
    """
    치조골 소실(RBL) 측정을 수행합니다.
    학습된 실제 모델이 부재한 경우 가짜 0.0 placeholder를 반환하지 않고 빈 결과를 반환합니다.
    """
    if model is None:
        return {'contours': [], 'status': 'UNIMPLEMENTED_MODEL'}
        
    # 향후 정식 모델 통합 시 실제 계측 로직 실행
    return {'contours': [], 'status': 'UNIMPLEMENTED_MODEL'}
