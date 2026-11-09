import io
import os
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
from core.pipeline import PanoramicPipeline


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
