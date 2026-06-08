import pytest
from evaluator import Evaluator

def test_evaluator_metrics():
    evaluator = Evaluator()
    
    # Mock data
    # Case 1: Perfect match
    evaluator.add_record(
        gt_rbl=10.0, pred_rbl=10.0,
        gt_landmarks={"mesial_cej": (10, 10)},
        pred_landmarks={"mesial_cej": (10, 10)}
    )
    
    # Case 2: Slight error
    evaluator.add_record(
        gt_rbl=30.0, pred_rbl=35.0,
        gt_landmarks={"mesial_cej": (100, 100)},
        pred_landmarks={"mesial_cej": (102, 102)}
    )
    
    # Check RBL MAE (Error sum = 0 + 5 = 5 / 2 = 2.5)
    mae_rbl = evaluator.calculate_rbl_mae()
    assert mae_rbl == 2.5
    
    # Check Coordinate MAE (Error sum: x=0, y=0, x=2, y=2 -> 4 / 4 coords = 1.0)
    mae_coords = evaluator.calculate_coordinate_mae()
    assert mae_coords == 1.0
    
    # Check R2 Score
    r2 = evaluator.calculate_rbl_r2_score()
    assert r2 < 1.0 and r2 > 0.0  # R2 should be reasonable
