from core.model_manager import ModelManager
from core.interfaces.dental_004 import init_004_model, run_super_resolution
from core.interfaces.dental_008 import init_008_model, run_tooth_segmentation, init_008_classifier, run_deciduous_classification
from core.interfaces.dental_002 import init_002_model, run_caries_detection
from core.interfaces.dental_003 import init_003_model, calculate_bone_loss
from modules.periapical_predictor import PeriapicalPredictorWrapper
from modules.restoration_predictor import RestorationPredictorWrapper
import numpy as np
import os

class PanoramicPipeline:
    """Dental Panoramic Reader 통합 파이프라인 오케스트레이터"""
    
    def __init__(self, use_004=False):
        self.manager = ModelManager()
        self.use_004 = use_004
        self.device = self.manager.device
        
        # 모델 초기화 및 매니저 등록
        # 필요 시점에만 GPU에 올리도록 처음엔 CPU로 로드
        
        # 004 모델 (선택적 구동)
        if self.use_004:
            model_004, config_004 = init_004_model()
            self.manager.register_model("004", model_004)
            self.config_004 = config_004
            
        # 008 모델 (필수: 치아 식별)
        model_008 = init_008_model()
        self.manager.register_model("008", model_008)
        
        # 008 분류기 (필수: 유치 판별)
        classifier_008 = init_008_classifier()
        self.manager.register_model("008_classifier", classifier_008)
        
        # 002 모델 (필수: 우식/매복 병소)
        model_002 = init_002_model()
        self.manager.register_model("002", model_002)
        
        # 003 모델 (필수: 치조골 소실)
        model_003 = init_003_model()
        if model_003 is not None:
            self.manager.register_model("003", model_003)
            
        # 012 모델 (필수: 치근단 병소 탐지)
        # Using submodule path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_012_path = os.path.join(base_dir, "modules", "Dental_012", "models", "best.pt")
        model_012 = PeriapicalPredictorWrapper(model_path=model_012_path)
        self.manager.register_model("012", model_012)
        
        # 013 모델 (필수: 수복물 분류)
        model_013_path = os.path.join(base_dir, "modules", "Dental_013", "models", "best.pt")
        model_013 = RestorationPredictorWrapper(model_path=model_013_path)
        self.manager.register_model("013", model_013)

    def run(self, image: np.ndarray) -> dict:
        """
        통합 진단 파이프라인을 실행합니다.
        """
        result_report = {}
        current_img = image.copy()
        
        # 1. 004 화질 개선 (Optional)
        if self.use_004:
            model_004 = self.manager.load_to_gpu("004")
            current_img = run_super_resolution(current_img, model_004, self.config_004, self.device)
            result_report['004_image'] = current_img
            
        # 1.5 008 이진 분류기 (유치 여부 판별)
        classifier_008 = self.manager.load_to_gpu("008_classifier")
        has_deciduous = run_deciduous_classification(current_img, classifier_008, self.device)
        result_report['has_deciduous'] = has_deciduous
            
        # 2. 008 치아 식별 및 마스킹
        model_008 = self.manager.load_to_gpu("008")
        tooth_roi_data = run_tooth_segmentation(current_img, model_008, self.device)
        result_report['008_tooth_data'] = tooth_roi_data
        
        # 3. 002 우식 및 병소 탐지
        model_002 = self.manager.load_to_gpu("002")
        caries_data = run_caries_detection(current_img, model_002)
        
        # [정합 로직] 002의 BBox(병소)가 008의 BBox(치아) 영역에 포함/교차되는지 판별하여 FDI 매핑
        mapped_lesions = self._map_lesions_to_fdi(caries_data, tooth_roi_data)
        result_report['002_lesions'] = mapped_lesions
        
        # 4. 003 치조골 소실 측정
        # 사용자 지시사항: 유치(has_deciduous)가 감지되면 치아 식별(008)과 병소 판독(002)은 진행하되 치조골 레벨 측정을 건너뜀.
        if has_deciduous:
            result_report['003_bone_loss'] = None
            print("Notice: Deciduous teeth detected. Skipping 003 Bone Loss Measurement.")
        else:
            model_003 = self.manager.load_to_gpu("003")
            if model_003 is not None:
                bone_loss_data = calculate_bone_loss(current_img, tooth_roi_data, model_003)
                result_report['003_bone_loss'] = bone_loss_data
                
        # 5. 012 치근단 병소 탐지 및 FDI 매칭
        model_012 = self.manager.load_to_gpu("012")
        if model_012 is not None:
            # 008에서 얻은 tooth_roi_data 형식을 012의 _match_fdi에 맞게 변환
            # (periapical_predictor expects a list of dicts: {"fdi": int, "contour": np.array})
            formatted_teeth_data_012 = []
            if tooth_roi_data and 'fdi_labels' in tooth_roi_data and 'contours' in tooth_roi_data:
                for fdi, contour in zip(tooth_roi_data['fdi_labels'], tooth_roi_data['contours']):
                    formatted_teeth_data_012.append({"fdi": fdi, "contour": contour})
            
            periapical_data = model_012.predict(current_img, teeth_data=formatted_teeth_data_012)
            result_report['012_periapical'] = periapical_data
            
        # 6. 013 수복물/보철물 분류
        model_013 = self.manager.load_to_gpu("013")
        if model_013 is not None:
            formatted_teeth_data_013 = []
            if tooth_roi_data and 'fdi_labels' in tooth_roi_data and 'boxes' in tooth_roi_data:
                for fdi, bbox in zip(tooth_roi_data['fdi_labels'], tooth_roi_data['boxes']):
                    formatted_teeth_data_013.append({"fdi": fdi, "bbox": bbox})
                    
            restoration_data = model_013.predict(current_img, teeth_data=formatted_teeth_data_013)
            result_report['013_restoration'] = restoration_data
            
        # GPU 캐시 비우기
        self.manager.clear_cache()
        
        return result_report

    def _map_lesions_to_fdi(self, caries_data, tooth_roi_data):
        """002 모듈의 병소 BBox와 008 모듈의 치아 BBox를 기하학적으로 매핑합니다."""
        mapped = []
        for i, lesion_box in enumerate(caries_data['boxes']):
            lesion_label = caries_data['labels'][i]
            best_iou = 0
            best_fdi = "Unknown"
            
            # 모든 치아 박스와 비교하여 가장 겹침(IoU)이 큰 치아 번호를 할당
            for j, tooth_box in enumerate(tooth_roi_data['boxes']):
                fdi = tooth_roi_data['fdi_labels'][j]
                iou = self._calculate_iou(lesion_box, tooth_box)
                if iou > best_iou:
                    best_iou = iou
                    best_fdi = fdi
                    
            mapped.append({
                'lesion_type': lesion_label,
                'box': lesion_box,
                'fdi': best_fdi
            })
            
        return mapped
        
    def _calculate_iou(self, boxA, boxB):
        # 단순 교차 영역 면적 비율 (Intersection over Area of Lesion)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        return interArea / float(boxAArea) if boxAArea > 0 else 0
