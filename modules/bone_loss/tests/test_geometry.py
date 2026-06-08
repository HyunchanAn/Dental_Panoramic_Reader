"""
Test Module for Geometry Utilities.
"""

import math
from utils.geometry import calculate_distance, calculate_rbl, pixel_to_mm

def test_calculate_distance():
    assert math.isclose(calculate_distance((0, 0), (3, 4)), 5.0)

def test_calculate_rbl_normal():
    # CEJ(0, 10), Crest(0, 12), Apex(0, 20)
    # Root: 10, Bone Loss: 2 => RBL = 20%
    rbl = calculate_rbl((0, 10), (0, 12), (0, 20))
    assert math.isclose(rbl, 20.0)

def test_calculate_rbl_clamped():
    # CEJ(0, 10), Crest(0, 8), Apex(0, 20)
    # Crest is coronal to CEJ => RBL = 0.0%
    rbl = calculate_rbl((0, 10), (0, 8), (0, 20))
    assert rbl == 0.0

if __name__ == "__main__":
    test_calculate_distance()
    test_calculate_rbl_normal()
    test_calculate_rbl_clamped()
    print("Geometry tests passed.")
