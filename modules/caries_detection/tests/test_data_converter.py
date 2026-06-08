import pytest
import sys
import os

# Add src to path to import data_converter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_converter import coco_to_yolo_bbox

def test_coco_to_yolo_bbox_normal():
    # Image size: 1000x1000
    # BBox: [100, 100, 200, 200] (x_min, y_min, w, h)
    # Center: (100+100, 100+100) = (200, 200)
    # Normalized: (200/1000, 200/1000, 200/1000, 200/1000) = (0.2, 0.2, 0.2, 0.2)
    bbox = [100, 100, 200, 200]
    img_w, img_h = 1000, 1000
    
    cx, cy, nw, nh = coco_to_yolo_bbox(bbox, img_w, img_h)
    
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.2)
    assert nw == pytest.approx(0.2)
    assert nh == pytest.approx(0.2)

def test_coco_to_yolo_bbox_clipping_negative():
    # BBox starts outside (negative)
    bbox = [-50, -50, 100, 100]
    img_w, img_h = 1000, 1000
    
    # cx = (-50 + 50) / 1000 = 0.0
    # cy = (-50 + 50) / 1000 = 0.0
    cx, cy, nw, nh = coco_to_yolo_bbox(bbox, img_w, img_h)
    
    assert cx == 0.0
    assert cy == 0.0
    assert nw == pytest.approx(0.1)
    assert nh == pytest.approx(0.1)

def test_coco_to_yolo_bbox_clipping_overflow():
    # BBox goes beyond image boundaries
    bbox = [950, 950, 100, 100]
    img_w, img_h = 1000, 1000
    
    # cx = (950 + 50) / 1000 = 1.0
    # cy = (950 + 50) / 1000 = 1.0
    cx, cy, nw, nh = coco_to_yolo_bbox(bbox, img_w, img_h)
    
    assert cx == 1.0
    assert cy == 1.0
    # nw = 100/1000 = 0.1 (Note: norm_w is also clipped to [0,1])
    assert nw == pytest.approx(0.1)
    assert nh == pytest.approx(0.1)

def test_coco_to_yolo_bbox_zero_size():
    bbox = [500, 500, 0, 0]
    img_w, img_h = 1000, 1000
    
    cx, cy, nw, nh = coco_to_yolo_bbox(bbox, img_w, img_h)
    
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert nw == 0.0
    assert nh == 0.0
