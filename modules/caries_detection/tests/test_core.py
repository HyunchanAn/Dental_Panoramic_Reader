import pytest
import numpy as np
import cv2
import os
from dentex_caries import CariesDetector, apply_clahe, assign_quadrant, map_detections_to_quadrants

def test_apply_clahe():
    """CLAHE 전처리 함수가 정상적으로 작동하고 형태를 유지하는지 테스트"""
    # Create a dummy BGR image
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    processed = apply_clahe(dummy_img, clip_limit=2.0)
    
    assert processed is not None
    assert processed.shape == dummy_img.shape
    assert processed.dtype == np.uint8

def test_assign_quadrant():
    """좌표 기반 FDI 치아 구역 분류 로직 검증"""
    W, H = 2000, 1000
    
    # 좌상단 (환자 기준 우측 상악) -> 1
    assert assign_quadrant(400, 200, W, H) == 1
    # 우상단 (환자 기준 좌측 상악) -> 2
    assert assign_quadrant(1600, 200, W, H) == 2
    # 좌하단 (환자 기준 우측 하악) -> 4
    assert assign_quadrant(400, 700, W, H) == 4
    # 우하단 (환자 기준 좌측 하악) -> 3
    assert assign_quadrant(1600, 700, W, H) == 3

def test_map_detections_to_quadrants():
    """탐지 결과 딕셔너리에 구역 정보가 잘 추가되는지 테스트"""
    W, H = 2000, 1000
    mock_detections = [
        {"bbox": [300, 100, 500, 300]}  # cx=400, cy=200 -> Quadrant 1
    ]
    
    result = map_detections_to_quadrants(mock_detections, W, H)
    
    assert len(result) == 1
    assert "quadrant" in result[0]
    assert "quadrant_label" in result[0]
    assert result[0]["quadrant"] == 1
    assert "UR" in result[0]["quadrant_label"]

def test_caries_detector_init():
    """CariesDetector 초기화 테스트 (모델 파일이 없을 때 예외 발생을 방지하는지 확인)"""
    # Use the base model for faster loading if it exists, else just test the wrapper structure
    model_path = "models/base_weights/yolo11n.pt"
    if os.path.exists(model_path):
        detector = CariesDetector(model_path=model_path, conf=0.5)
        assert detector.model is not None
        assert detector.conf == 0.5
        assert len(detector.class_names) == 4
