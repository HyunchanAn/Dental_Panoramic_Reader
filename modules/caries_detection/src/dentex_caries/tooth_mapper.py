import numpy as np

# FDI 기준 4개 구역 정의
# 파노라마 영상에서 이미지 좌측 = 환자 우측(Right), 이미지 우측 = 환자 좌측(Left)
QUADRANT_LABELS = {
    1: "상악 우측 (UR)",  # Upper Right
    2: "상악 좌측 (UL)",  # Upper Left
    3: "하악 좌측 (LL)",  # Lower Left
    4: "하악 우측 (LR)",  # Lower Right
}

def assign_quadrant(cx, cy, img_w, img_h):
    """
    파노라마 영상의 좌표를 기반으로 FDI 구역(Quadrant) 번호를 반환합니다.
    정확한 치아 번호 부여는 전용 Segmentation 모델이 필요하므로,
    신뢰 가능한 구역(1~4사분면) 수준까지만 추정합니다.

    교합면(Occlusal Plane)은 파노라마 영상 특성상 이미지 상단에서
    약 55% 지점에 위치하는 것으로 보정합니다.
    """
    mid_x = img_w / 2
    occlusal_y = img_h * 0.55  # 교합면: 이미지 상단에서 55% 지점

    is_upper = cy < occlusal_y
    is_left_in_image = cx < mid_x  # 이미지 기준 좌측 = 환자 우측

    if is_upper and is_left_in_image:
        return 1  # 상악 우측 (UR)
    elif is_upper and not is_left_in_image:
        return 2  # 상악 좌측 (UL)
    elif not is_upper and not is_left_in_image:
        return 3  # 하악 좌측 (LL)
    else:
        return 4  # 하악 우측 (LR)


def map_detections_to_quadrants(detections, img_w, img_h):
    """
    탐지 결과 목록에 구역(Quadrant) 번호 및 레이블을 추가합니다.
    """
    for det in detections:
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        quadrant = assign_quadrant(cx, cy, img_w, img_h)
        det["quadrant"] = quadrant
        det["quadrant_label"] = QUADRANT_LABELS[quadrant]

    return detections


if __name__ == "__main__":
    # 간단한 동작 테스트
    W, H = 2000, 1000
    test_cases = [
        {"box": (400, 200), "expected": "상악 우측 (UR)"},   # 이미지 좌상단
        {"box": (1600, 200), "expected": "상악 좌측 (UL)"},  # 이미지 우상단
        {"box": (400, 700), "expected": "하악 우측 (LR)"},   # 이미지 좌하단
        {"box": (1600, 700), "expected": "하악 좌측 (LL)"},  # 이미지 우하단
    ]

    print("구역 분류 테스트 결과:")
    for case in test_cases:
        cx, cy = case["box"]
        q = assign_quadrant(cx, cy, W, H)
        label = QUADRANT_LABELS[q]
        status = "OK" if label == case["expected"] else "FAIL"
        print(f"  [{status}] ({cx}, {cy}) -> {label}  (기대: {case['expected']})")
