from core.model_manager import ModelManager, get_logger
from core.interfaces.dental_004 import init_004_model, run_super_resolution
from core.interfaces.dental_008 import init_008_model, run_tooth_segmentation, init_008_classifier, run_deciduous_classification
from core.interfaces.dental_002 import init_002_model, run_caries_detection
from core.interfaces.dental_003 import init_003_model, calculate_bone_loss
from modules.periapical_predictor import PeriapicalPredictorWrapper
from modules.restoration_predictor import RestorationPredictorWrapper
from modules.osteoporosis_predictor import OsteoporosisPredictorWrapper
import numpy as np
import os
import json
import subprocess

_logger = get_logger("PanoramicPipeline")


def get_submodule_shas(base_dir: str) -> dict:
    """각 내부 서브모듈(modules/Dental_*)의 현재 Git 커밋 SHA를 수집합니다."""
    shas = {}
    modules_dir = os.path.join(base_dir, "modules")
    if not os.path.isdir(modules_dir):
        return shas
    for name in sorted(os.listdir(modules_dir)):
        sub_path = os.path.join(modules_dir, name)
        if name.startswith("Dental_") and os.path.isdir(sub_path):
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=sub_path,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    shas[name] = res.stdout.strip()
            except Exception:
                pass
    return shas


def load_clinical_criteria(criteria_path=None):
    """임상 기준표 SSOT(clinical_criteria_v1.json)를 로드하거나 기본값을 제공합니다."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.environ.get("CLINICAL_CRITERIA_PATH")

    candidate_paths = [
        p
        for p in [
            criteria_path,
            env_path,
            os.path.join(base_dir, "plans_replies_walkthrough", "clinical_criteria_v1.json"),
        ]
        if p
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _logger.warning(f"Error loading criteria from {p}: {e}")

    return {
        "supernumerary_policy": "SEPARATE_ENTITY",
        "supernumerary_naming": "FDI Quadrant Code + 9 (예: #19, #29) 또는 SN-1",
        "periapical_threshold_definite": 0.50,
        "periapical_threshold_review": 0.25,
        "periapical_anatomical_guard": True,
        "restoration_classes": ["Inlay/Onlay", "Crown/Bridge", "Implant", "Endodontic (RCT)"],
        "restoration_mapping": {
            "Amalgam / Resin / Inlay": "Inlay/Onlay",
            "Gold / PFM / Zirconia Crown": "Crown/Bridge",
            "Implant Fixture / Abutment": "Implant",
            "Root Canal Filling / Post": "Endodontic (RCT)",
        },
    }


class PanoramicPipeline:
    """Dental Panoramic Reader 통합 파이프라인 오케스트레이터"""

    def __init__(self, use_004=False, criteria_path=None):
        self.manager = ModelManager()
        self.logger = _logger
        self.use_004 = use_004
        self.device = self.manager.device
        self.clinical_criteria = load_clinical_criteria(criteria_path)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.module_versions = get_submodule_shas(base_dir)

        if self.use_004:
            model_004, config_004 = init_004_model()
            self.manager.register_model("004", model_004)
            self.config_004 = config_004

        model_008 = init_008_model()
        self.manager.register_model("008", model_008)

        classifier_008 = init_008_classifier()
        self.manager.register_model("008_classifier", classifier_008)

        model_002 = init_002_model()
        self.manager.register_model("002", model_002)

        model_003 = init_003_model()
        if model_003 is not None:
            self.manager.register_model("003", model_003)

        model_012_candidates = [
            os.path.join(base_dir, "modules", "Dental_012", "models", "best.pt"),
            os.path.join(base_dir, "modules", "Dental_012", "models", "best.onnx"),
        ]
        model_012_path = next(
            (p for p in model_012_candidates if os.path.exists(p)),
            model_012_candidates[1],
        )
        model_012 = PeriapicalPredictorWrapper(model_path=model_012_path)
        self.manager.register_model("012", model_012)

        model_013_path = os.path.join(
            base_dir, "modules", "Dental_013", "models", "best_restoration_model.onnx"
        )
        if not os.path.exists(model_013_path):
            model_013_path = os.path.join(
                base_dir, "modules", "Dental_013", "models", "best_restoration_model.pth"
            )
        model_013 = RestorationPredictorWrapper(model_path=model_013_path)
        self.manager.register_model("013", model_013)

        model_014_path = os.path.join(
            base_dir, "modules", "Dental_014", "models", "best.pt"
        )
        model_014 = OsteoporosisPredictorWrapper(
            model_path=model_014_path, device=str(self.device)
        )
        self.manager.register_model("014", model_014)

    def run(self, image: np.ndarray) -> dict:
        """
        통합 진단 파이프라인을 실행합니다.
        """
        result_report = {
            "clinical_criteria_version": "v1.0",
            "module_versions": self.module_versions,
        }
        current_img = image.copy()

        # 1. 004 화질 개선 (Optional)
        if self.use_004:
            model_004 = self.manager.load_to_gpu("004")
            current_img = run_super_resolution(
                current_img, model_004, self.config_004, self.device
            )
            result_report["004_image"] = current_img

        # 1.5 008 이진 분류기 (유치 여부 판별)
        classifier_008 = self.manager.load_to_gpu("008_classifier")
        has_deciduous = run_deciduous_classification(
            current_img, classifier_008, self.device
        )
        result_report["has_deciduous"] = has_deciduous

        # 2. 008 치아 식별 및 마스킹 (임상 기준 SSOT 연동)
        model_008 = self.manager.load_to_gpu("008")
        tooth_roi_data = run_tooth_segmentation(
            current_img, model_008, self.device, criteria=self.clinical_criteria
        )
        result_report["008_tooth_data"] = tooth_roi_data

        # 3. 002 우식 및 병소 탐지 (2-Stage 치아 패치 연계)
        model_002 = self.manager.load_to_gpu("002")
        tooth_boxes = tooth_roi_data.get("boxes", []) if tooth_roi_data else []
        caries_data = run_caries_detection(
            current_img, model_002, tooth_boxes=tooth_boxes
        )

        mapped_lesions = self._map_lesions_to_fdi(caries_data, tooth_roi_data)
        result_report["002_lesions"] = mapped_lesions

        # 4. 003 치조골 소실 측정
        if has_deciduous:
            result_report["003_bone_loss"] = None
            self.logger.info(
                "Notice: Deciduous teeth detected. Skipping 003 Bone Loss Measurement."
            )
        else:
            model_003 = self.manager.load_to_gpu("003")
            if model_003 is not None:
                bone_loss_data = calculate_bone_loss(
                    current_img, tooth_roi_data, model_003
                )
                result_report["003_bone_loss"] = bone_loss_data

        # 5. 012 치근단 병소 탐지 및 FDI 매칭
        model_012 = self.manager.load_to_gpu("012")
        if model_012 is not None:
            formatted_teeth_data_012 = []
            if tooth_roi_data and "fdi_labels" in tooth_roi_data:
                fdis = tooth_roi_data["fdi_labels"]
                boxes = tooth_roi_data.get("boxes", [])
                contours = tooth_roi_data.get("contours", [])
                for i in range(len(fdis)):
                    fdi = fdis[i]
                    bbox = boxes[i] if i < len(boxes) else None
                    contour = contours[i] if i < len(contours) else None
                    formatted_teeth_data_012.append(
                        {"fdi": fdi, "contour": contour, "bbox": bbox}
                    )

            periapical_data = model_012.predict(
                current_img,
                teeth_data=formatted_teeth_data_012,
                criteria=self.clinical_criteria,
            )
            result_report["012_periapical"] = periapical_data

        # 6. 013 수복물/보철물 분류
        model_013 = self.manager.load_to_gpu("013")
        if model_013 is not None:
            formatted_teeth_data_013 = []
            if (
                tooth_roi_data
                and "fdi_labels" in tooth_roi_data
                and "boxes" in tooth_roi_data
            ):
                for fdi, bbox in zip(
                    tooth_roi_data["fdi_labels"], tooth_roi_data["boxes"]
                ):
                    formatted_teeth_data_013.append({"fdi": fdi, "bbox": bbox})

            restoration_data = model_013.predict(
                current_img,
                teeth_data=formatted_teeth_data_013,
                criteria=self.clinical_criteria,
            )
            result_report["013_restoration"] = restoration_data

        # 7. 009 매복 제3대구치 분석
        try:
            from modules.Dental_009.src.analyzer import ImpactedToothAnalyzer

            analyzer_009 = ImpactedToothAnalyzer()
            impacted_results = []

            pairs = [(18, 17), (28, 27), (38, 37), (48, 47)]
            tooth_dict = {}
            if (
                tooth_roi_data
                and "fdi_labels" in tooth_roi_data
                and "boxes" in tooth_roi_data
            ):
                for fdi, box in zip(
                    tooth_roi_data["fdi_labels"], tooth_roi_data["boxes"]
                ):
                    tooth_dict[fdi] = box

            for m3_fdi, m2_fdi in pairs:
                if m3_fdi in tooth_dict:
                    m3_box = tooth_dict[m3_fdi]
                    m2_box = tooth_dict.get(m2_fdi)

                    m3_pts = np.array(
                        [
                            [m3_box[0], m3_box[1]],
                            [m3_box[2], m3_box[1]],
                            [m3_box[2], m3_box[3]],
                            [m3_box[0], m3_box[3]],
                        ],
                        dtype=np.int32,
                    )

                    if m2_box is not None:
                        m2_pts = np.array(
                            [
                                [m2_box[0], m2_box[1]],
                                [m2_box[2], m2_box[1]],
                                [m2_box[2], m2_box[3]],
                                [m2_box[0], m2_box[3]],
                            ],
                            dtype=np.int32,
                        )
                        res = analyzer_009.analyze_impacted_tooth(
                            m3_pts, m3_fdi, m2_pts
                        )
                    else:
                        res = {
                            "fdi": m3_fdi,
                            "angle_diff": 0.0,
                            "winters_class": "Isolated / Impacted",
                            "eruption_status": "Fully Impacted",
                        }

                    is_normally_erupted = (
                        res.get("eruption_status") == "Fully Erupted"
                        and res.get("winters_class") == "Vertical"
                    )
                    if not is_normally_erupted:
                        res["box"] = (
                            m3_box.tolist()
                            if hasattr(m3_box, "tolist")
                            else list(m3_box)
                        )
                        impacted_results.append(res)

            result_report["009_impacted"] = impacted_results
        except Exception as e:
            self.logger.warning(f"Dental_009 impaction analysis warning: {e}")
            result_report["009_impacted"] = []

        # 8. 014 골다공증 스크리닝
        model_014 = self.manager.load_to_gpu("014")
        if model_014 is not None:
            osteo_data = model_014.predict(current_img)
            result_report["014_osteoporosis"] = osteo_data
        else:
            result_report["014_osteoporosis"] = None

        self.manager.clear_cache()

        return result_report

    def _map_lesions_to_fdi(self, caries_data, tooth_roi_data):
        mapped = []
        if not caries_data or "boxes" not in caries_data or not caries_data["boxes"]:
            return mapped

        c_boxes = caries_data["boxes"]
        c_labels = caries_data["labels"]
        c_scores = caries_data.get("scores", [])

        t_boxes = tooth_roi_data.get("boxes", [])
        t_fdi = tooth_roi_data.get("fdi_labels", [])

        for i, c_box in enumerate(c_boxes):
            best_fdi = "Unknown"
            max_inter = 0
            cx1, cy1, cx2, cy2 = c_box
            raw_conf = float(c_scores[i]) if i < len(c_scores) else 0.5

            for j, t_box in enumerate(t_boxes):
                tx1, ty1, tx2, ty2 = t_box

                ix1 = max(cx1, tx1)
                iy1 = max(cy1, ty1)
                ix2 = min(cx2, tx2)
                iy2 = min(cy2, ty2)

                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter > max_inter:
                        max_inter = inter
                        best_fdi = t_fdi[j]

            mapped.append(
                {
                    "lesion_type": c_labels[i],
                    "confidence": raw_conf,
                    "fdi": best_fdi,
                    "box": c_box,
                }
            )

        return mapped
