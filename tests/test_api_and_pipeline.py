import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import api_server
from core.periodontal_risk_gate import (
    build_structured_vision_llm_payload,
    verify_10_case_benchmark_vision_json_guard,
)
from core.pipeline import PanoramicPipeline
from modules.periapical_predictor import PeriapicalPredictorWrapper


def _make_png_bytes(width: int = 120, height: int = 80) -> bytes:
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_dicom_bytes(width: int = 64, height: int = 64) -> bytes:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientName = "Hong^GilDong"
    ds.PatientID = "PHI-12345"
    ds.PatientBirthDate = "19900101"
    ds.Rows = height
    ds.Columns = width
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    pixels = np.arange(width * height, dtype=np.uint8).reshape((height, width))
    ds.PixelData = pixels.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    return buf.getvalue()


def test_fail_closed_auth_when_token_unset_or_invalid(monkeypatch):
    client = TestClient(api_server.app)
    png_bytes = _make_png_bytes()

    # 1. Fail-closed when DPR_API_TOKEN is unset
    monkeypatch.delenv("DPR_API_TOKEN", raising=False)
    resp_unset = client.post(
        "/api/v1/infer",
        files={"file": ("pano.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp_unset.status_code == 401

    # 2. Fail-closed when DPR_API_TOKEN is set but header is missing or mismatched
    monkeypatch.setenv("DPR_API_TOKEN", "unit-test-secret-token")
    resp_missing = client.post(
        "/api/v1/infer",
        files={"file": ("pano.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp_missing.status_code == 401

    resp_wrong = client.post(
        "/api/v1/infer",
        headers={"X-API-Token": "wrong-token"},
        files={"file": ("pano.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp_wrong.status_code == 401

    # 3. Succeeds when valid token is provided
    mock_raw = {
        "008_tooth_data": {
            "fdi_status": "FDI_ACTIVE",
            "fdi_labels": [11, 21],
            "boxes": [[40, 10, 55, 40], [60, 10, 75, 40]],
            "scores": [0.95, 0.94],
        },
        "002_lesions": [],
        "003_bone_loss": {"contours": []},
        "011_age": {"predicted_age": 24.5},
        "012_periapical": {"lesions": []},
        "013_restoration": {"results": []},
        "009_impacted": [],
        "014_osteoporosis": {"pred_class": "Normal", "probabilities": {"Normal": 0.9}},
    }
    with patch.object(api_server, "get_pipeline") as mock_get_pipe:
        mock_get_pipe.return_value.run.return_value = mock_raw
        resp_ok = client.post(
            "/api/v1/infer",
            headers={"Authorization": "Bearer unit-test-secret-token"},
            files={"file": ("pano.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert resp_ok.status_code == 200
        body = resp_ok.json()
        assert body["patientId"].startswith("ANON-")
        assert body["patientId"] != "PATIENT-DEMO"
        assert body["findings"]["ageEstimation"]["predictedAge"] == pytest.approx(24.5)
        assert body["findings"]["ageEstimation"]["status"] == "ACTIVE_INFERENCE"


def test_in_memory_dicom_decoding_and_phi_stripping():
    import pydicom

    dcm_bytes = _make_dicom_bytes(48, 32)
    ds_check = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)
    ds_stripped = api_server.strip_phi_tags_in_memory(ds_check)
    assert str(ds_stripped.PatientName) == "ANONYMIZED"
    assert str(ds_stripped.PatientID) == "ANONYMIZED"
    assert str(ds_stripped.PatientBirthDate) == "ANONYMIZED"

    rgb_arr, is_dicom = api_server.decode_uploaded_radiograph_in_memory(
        dcm_bytes, filename="sample.dcm", content_type="application/dicom"
    )
    assert is_dicom is True
    assert rgb_arr.shape == (32, 48, 3)
    assert rgb_arr.dtype == np.uint8


def test_health_endpoint_readiness_and_degraded_states(monkeypatch):
    client = TestClient(api_server.app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("HEALTHY", "DEGRADED", "UNHEALTHY")

    mod_ids = {m["id"] for m in data["modules"]}
    assert "Dental_011" in mod_ids
    for mod in data["modules"]:
        assert isinstance(mod["configured"], bool)
        assert isinstance(mod["loaded"], bool)
        assert isinstance(mod["ready"], bool)

    # Simulate missing Dental_008 weights -> UNHEALTHY
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    resp_unhealthy = client.get("/api/v1/health")
    assert resp_unhealthy.status_code == 200
    assert resp_unhealthy.json()["status"] == "UNHEALTHY"


def test_ssot_report_execution_status_and_summary_guard():
    raw_partial = {
        "has_deciduous": True,
        "008_tooth_data": {"fdi_status": "FDI_UNAVAILABLE", "boxes": [], "fdi_labels": []},
        "003_bone_loss": None,
        "011_age": {"error": "Model not loaded"},
    }
    report = api_server.format_to_ssot_report(
        raw_partial,
        1000,
        500,
        "child_pano.png",
        image_hash="abcdef1234567890abcdef1234567890",
    )
    assert report["patientId"] == "ANON-ABCDEF123456"
    exec_status = report["findings"]["moduleExecutionStatus"]
    assert exec_status["Dental_003"]["status"] == "skipped"
    assert exec_status["Dental_011"]["status"] == "unavailable"
    assert "특이 소견이 없습니다" not in report["summary"]
    assert "보류" in report["summary"] or "미실행" in report["summary"]


def test_pipeline_registers_011_and_tracks_module_execution():
    pipe = PanoramicPipeline(use_004=False)
    assert "011" in pipe.manager.models


def test_missing_data_gate_and_clinical_metadata_risk_assessment(monkeypatch):
    client = TestClient(api_server.app)
    monkeypatch.setenv("DPR_API_TOKEN", "gate-test-token")
    png_bytes = _make_png_bytes(200, 100)

    mock_raw = {
        "008_tooth_data": {
            "fdi_status": "FDI_ACTIVE",
            "fdi_labels": [16, 45],
            "boxes": [[20, 10, 45, 45], [50, 55, 75, 90]],
            "scores": [0.96, 0.94],
        },
        "002_lesions": [],
        "003_bone_loss": {
            "patient_stage": "Stage II",
            "contours": [
                {
                    "fdi": 16,
                    "pts": [[20, 20], [45, 20], [45, 32], [20, 32]],
                    "rbl_percent": 26.5,
                    "loss_mm": 3.4,
                    "mm_calibrated": False,
                    "stage": "Stage II",
                    "confidence": 0.89,
                }
            ],
            "metrics": [{"tooth_number": 16, "rbl_percent": 26.5, "loss_mm": 3.4}],
        },
        "013_restoration": {
            "results": [
                {
                    "fdi": 45,
                    "clinical_category": "Endodontic (RCT)",
                    "raw_class": "Root Canal Treatment",
                    "confidence": 0.91,
                    "bbox": [52, 60, 72, 88],
                }
            ]
        },
        "012_periapical": {
            "lesions": [
                {
                    "bbox": [54, 84, 70, 96],
                    "confidence": 0.72,
                    "clinical_status": "DEFINITE",
                    "fdi": 45,
                    "rct_associated": True,
                    "lesion_subtype": "post_rct_periapical_lesion",
                    "isolated_anatomical_suspicion": False,
                }
            ]
        },
        "009_impacted": [],
        "011_age": {"predicted_age": 44.0},
        "014_osteoporosis": None,
    }

    with patch.object(api_server, "get_pipeline") as mock_get_pipe:
        mock_get_pipe.return_value.run.return_value = mock_raw

        # 1) Radiograph-only upload -> Missing Data Gate withholds definitive Grade & Risk
        resp_only_img = client.post(
            "/api/v1/infer",
            headers={"X-API-Token": "gate-test-token"},
            files={"file": ("pano.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert resp_only_img.status_code == 200
        body_incomplete = resp_only_img.json()
        gate_inc = body_incomplete["findings"]["periodontalRiskAssessment"]
        assert gate_inc["assessmentStatus"] == "INCOMPLETE_CLINICAL_DATA"
        assert "psiScore_or_bopPercent" in gate_inc["missingClinicalFields"]
        assert "smokingStatus" in gate_inc["missingClinicalFields"]
        assert "diabetesStatus" in gate_inc["missingClinicalFields"]
        assert gate_inc["grade"] is None
        assert gate_inc["riskLevel"] is None

        # Verify Dental_013 -> teeth[].restorationTags and Dental_012 -> rctAssociated propagation
        tooth_45 = next(t for t in body_incomplete["findings"]["teeth"] if t["toothNumber"] == 45)
        assert "Endodontic (RCT)" in tooth_45["restorationTags"]
        pa_0 = body_incomplete["findings"]["periapicalLesions"][0]
        assert pa_0["rctAssociated"] is True
        assert pa_0["lesionSubtype"] == "post_rct_periapical_lesion"

        # 2) Radiograph + Complete Clinical Metadata -> COMPLETE Stage/Grade/Risk
        clinical_json = json.dumps(
            {
                "psiScore": 4,
                "bopPercent": 30.0,
                "plaqueIndex": 35.0,
                "smokingStatus": "current_heavy",
                "diabetesStatus": "controlled",
                "ageYears": 46,
            }
        )
        resp_complete = client.post(
            "/api/v1/infer",
            headers={"X-API-Token": "gate-test-token"},
            files={"file": ("pano.png", io.BytesIO(png_bytes), "image/png")},
            data={"clinical_metadata": clinical_json},
        )
        assert resp_complete.status_code == 200
        body_comp = resp_complete.json()
        gate_comp = body_comp["findings"]["periodontalRiskAssessment"]
        assert gate_comp["assessmentStatus"] == "COMPLETE"
        assert gate_comp["missingClinicalFields"] == []
        assert gate_comp["stage"] == "Stage III"
        assert gate_comp["grade"] == "Grade C"
        assert gate_comp["riskLevel"] == "HIGH"

        llm_payload = build_structured_vision_llm_payload(body_comp)
        assert "subgingival_calculus" in llm_payload["strict_non_inferable_from_panorama"]
        assert llm_payload["quantified_radiographic_evidence"]["periapical_lesions"][0]["rct_associated"] is True


def test_hitl_save_and_get_roundtrip(monkeypatch, tmp_path: Path):
    client = TestClient(api_server.app)
    monkeypatch.setenv("DPR_API_TOKEN", "hitl-secret")
    monkeypatch.setenv("DPR_HITL_STORAGE_DIR", str(tmp_path))

    golden_payload = {
        "schema_version": "dental_core.panorama_hitl.v1",
        "document_id": "HITL-UNITTEST0001",
        "hitl_status": "verified_golden",
        "annotator": "clinician_test",
        "source_model": "Dental_008+Dental_015_HiTL",
        "image_meta": {
            "filename": "sample_pano.png",
            "width": 1000,
            "height": 500,
            "sha256_hash": "deadbeef",
            "source_format": "PNG",
            "pixel_spacing_mm": None,
            "is_calibrated": False,
            "midline_x": 500.0,
        },
        "teeth": [
            {
                "instance_id": "tooth-36",
                "fdi": 36,
                "bbox_xyxy": [680.0, 260.0, 760.0, 390.0],
                "polygon_px": [[680.0, 260.0], [760.0, 260.0], [760.0, 390.0], [680.0, 390.0]],
                "confidence": 0.98,
                "uncertain": False,
                "is_supernumerary": False,
                "restoration_tags": ["Crown/Bridge"],
                "clinician_corrected": True,
                "original_fdi": 37,
            }
        ],
        "lesions": [
            {
                "instance_id": "lesion-c1",
                "category": "caries",
                "label": "Caries",
                "bbox_xyxy": [700.0, 270.0, 735.0, 305.0],
                "fdi": 36,
                "confidence": 1.0,
                "clinical_status": "DEFINITE",
                "clinician_corrected": True,
            }
        ],
        "bone_loss": [],
        "missing_fdi": [16],
    }

    # Unauthorized without token
    assert client.post("/api/v1/hitl/save", json=golden_payload).status_code == 401

    resp_save = client.post(
        "/api/v1/hitl/save",
        headers={"X-API-Token": "hitl-secret"},
        json=golden_payload,
    )
    assert resp_save.status_code == 200
    assert resp_save.json()["status"] == "saved"
    assert resp_save.json()["document_id"] == "HITL-UNITTEST0001"

    resp_get = client.get(
        "/api/v1/hitl/HITL-UNITTEST0001",
        headers={"X-API-Token": "hitl-secret"},
    )
    assert resp_get.status_code == 200
    loaded = resp_get.json()
    assert loaded["teeth"][0]["fdi"] == 36
    assert loaded["teeth"][0]["original_fdi"] == 37
    assert loaded["lesions"][0]["category"] == "caries"


def test_periapical_wrapper_rct_tie_breaker_and_10_cases_benchmark_guard(tmp_path: Path):
    wrapper = PeriapicalPredictorWrapper(model_path="nonexistent.onnx")
    teeth_data = [
        {"fdi": 44, "bbox": [200, 350, 240, 480], "contour": None},
        {"fdi": 45, "bbox": [250, 350, 290, 480], "contour": None},
    ]
    lesions = [
        {
            "bbox": [231, 470, 261, 500],  # Between #44 and #45; #45 has RCT
            "confidence": 0.42,
            "clinical_status": "REVIEW_REQUIRED",
            "fdi": None,
            "isolated_anatomical_suspicion": False,
        }
    ]
    wrapper._match_fdi(lesions, teeth_data, anatomical_guard=True, rct_fdis=[45])
    assert lesions[0]["fdi"] == 45
    assert lesions[0]["rct_associated"] is True
    assert lesions[0]["lesion_subtype"] == "post_rct_periapical_lesion"
    assert lesions[0]["clinical_status"] == "DEFINITE"

    cand_bench = Path(ROOT_DIR).parent / "Dental_000" / "reports" / "10_cases_quantitative_benchmark.json"
    if not cand_bench.exists():
        cand_bench = tmp_path / "synthetic_10_cases.json"
        cand_bench.write_text(
            json.dumps(
                {
                    "summary": [
                        {"case": f"case_{i}.png", "module": "Composite" if i == 0 else "Dental_008"}
                        for i in range(10)
                    ]
                }
            ),
            encoding="utf-8",
        )
    bench_res = verify_10_case_benchmark_vision_json_guard(cand_bench)
    assert bench_res["total_cases"] >= 10
    assert bench_res["all_radiograph_only_gated"] is True
