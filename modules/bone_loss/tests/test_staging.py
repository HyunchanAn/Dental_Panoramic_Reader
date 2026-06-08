"""
Test Module for Staging Logic.
"""

from services.staging import determine_patient_stage, Stage, Extent

def test_staging_stage_i_localized():
    metrics = [{"tooth": 11, "max_rbl": 10.0}, {"tooth": 12, "max_rbl": 5.0}]
    stage, extent = determine_patient_stage(metrics)
    assert stage == Stage.STAGE_I
    assert extent == Extent.LOCALIZED

def test_staging_stage_iv_generalized():
    # 4개 치아 중 2개가 15% 이상(50%), 최고 RBL 35% + severe complexity
    metrics = [
        {"tooth": 11, "max_rbl": 35.0},
        {"tooth": 12, "max_rbl": 20.0},
        {"tooth": 21, "max_rbl": 5.0},
        {"tooth": 22, "max_rbl": 5.0}
    ]
    stage, extent = determine_patient_stage(metrics, has_severe_complexity=True)
    assert stage == Stage.STAGE_IV
    assert extent == Extent.GENERALIZED

if __name__ == "__main__":
    test_staging_stage_i_localized()
    test_staging_stage_iv_generalized()
    print("Staging tests passed.")
