import numpy as np
from typing import List, Dict, Any

def verify_missing_teeth(
    detected_fdi_labels: List[int],
    boxes: List[List[float]],
    img_width: int,
    img_height: int,
    midline_x: float
) -> Dict[str, Any]:
    """
    2차 갭 비율(Gap Distance Ratio) 및 최후방 구치 예외 처리를 적용한 결손치 검증 알고리즘.
    """
    # 전 영구치 32개 집합
    all_permanent = set([
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48
    ])
    
    detected_set = set(detected_fdi_labels)
    candidate_missing = sorted(list(all_permanent - detected_set))
    
    if not boxes or len(boxes) == 0:
        return {
            "verified_missing": candidate_missing,
            "uncertain_missing": [],
            "details": []
        }
        
    # FDI별 치아 box 매핑
    fdi_to_box = {}
    for fdi, box in zip(detected_fdi_labels, boxes):
        fdi_to_box[fdi] = box

    # 평균 치아 너비 계산
    widths = [b[2] - b[0] for b in boxes]
    avg_tooth_width = np.mean(widths) if len(widths) > 0 else 50.0

    verified_missing = []
    uncertain_missing = []
    details = []

    posterior_teeth = {18, 28, 38, 48, 17, 27, 37, 47}

    for missing_fdi in candidate_missing:
        # 1. 최후방 구치인 경우 편측 인접치 부재 예외 처리
        if missing_fdi in posterior_teeth:
            uncertain_missing.append(missing_fdi)
            details.append({
                "fdi": missing_fdi,
                "status": "UNCERTAIN_DISTAL_MISSING",
                "reason": "Outer distal adjacent tooth absent (Posterior region)"
            })
            continue

        # 2. 양측 인접 치아 탐색 (근심/원심)
        # 예: 16번 결손 후보 -> 인접한 15번, 17번 탐색
        quadrant = missing_fdi // 10
        pos = missing_fdi % 10
        
        mesial_fdi = missing_fdi - 1 if pos > 1 else missing_fdi + 10 # 정중선 방향
        distal_fdi = missing_fdi + 1 if pos < 8 else None            # 후방 방향

        left_box = fdi_to_box.get(mesial_fdi)
        right_box = fdi_to_box.get(distal_fdi) if distal_fdi else None

        if left_box is not None and right_box is not None:
            # 좌우 인접치 X-중심간 거리 계산
            left_center_x = (left_box[0] + left_box[2]) / 2.0
            right_center_x = (right_box[0] + right_box[2]) / 2.0
            gap_distance = abs(right_center_x - left_center_x)
            r_gap = gap_distance / avg_tooth_width if avg_tooth_width > 0 else 0.0

            if r_gap > 0.65:
                verified_missing.append(missing_fdi)
                details.append({
                    "fdi": missing_fdi,
                    "status": "VERIFIED_MISSING",
                    "r_gap": round(float(r_gap), 3),
                    "gap_distance": round(float(gap_distance), 1)
                })
            else:
                uncertain_missing.append(missing_fdi)
                details.append({
                    "fdi": missing_fdi,
                    "status": "UNCERTAIN_DRIFT_OR_IMPACTION",
                    "r_gap": round(float(r_gap), 3)
                })
        else:
            # 인접치 하나가 부족한 경우 1차 유보
            uncertain_missing.append(missing_fdi)
            details.append({
                "fdi": missing_fdi,
                "status": "UNCERTAIN_INSUFFICIENT_ADJACENT"
            })

    return {
        "verified_missing": verified_missing,
        "uncertain_missing": uncertain_missing,
        "details": details
    }
