import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
impacted_src_path = os.path.join(BASE_DIR, "modules", "Dental_009", "src")
if impacted_src_path not in sys.path:
    sys.path.insert(0, impacted_src_path)

from analyzer import ImpactedToothAnalyzer
from .base_predictor import BasePanoramicPredictor

class ImpactedToothPredictorWrapper(BasePanoramicPredictor):
    def __init__(self):
        self.analyzer = None
        self.load_model()

    def load_model(self, *args, **kwargs) -> None:
        self.analyzer = ImpactedToothAnalyzer()

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        """
        image 파라미터는 호환성을 위해 유지하지만, 실제 연산은 kwargs로 전달된 teeth_data를 기반으로 합니다.
        teeth_data: 008 모듈에서 반환된 [{'fdi': int, 'contour': list}, ...] 형태의 리스트
        """
        teeth_data = kwargs.get("teeth_data", [])
        
        if not teeth_data:
            return {
                "module_name": "Dental_009_impacted_tooth",
                "impacted_analysis": []
            }

        # FDI 번호를 key로, contour를 value로 갖는 딕셔너리 생성
        teeth_dict = {t["fdi"]: np.array(t["contour"], dtype=np.int32) for t in teeth_data if t["contour"] is not None}
        
        impacted_analysis = []
        
        # 제3대구치 후보군
        third_molars = [18, 28, 38, 48]
        
        for tm_fdi in third_molars:
            if tm_fdi in teeth_dict:
                # 인접 제2대구치 번호
                adj_fdi = tm_fdi - 1
                if adj_fdi in teeth_dict:
                    tm_contour = teeth_dict[tm_fdi]
                    adj_contour = teeth_dict[adj_fdi]
                    
                    analysis = self.analyzer.analyze_impacted_tooth(tm_contour, tm_fdi, adj_contour)
                    
                    if "error" not in analysis:
                        # 분석 결과에 렌더링을 위한 좌표(장축) 정보를 함께 담아줍니다.
                        tc, tvec = self.analyzer.get_long_axis_vector(tm_contour)
                        ac, avec = self.analyzer.get_long_axis_vector(adj_contour)
                        
                        analysis["long_axis"] = {
                            "target": {"center": tc, "vector": tvec},
                            "adjacent": {"center": ac, "vector": avec}
                        }
                        
                        impacted_analysis.append(analysis)
        
        return {
            "module_name": "Dental_009_impacted_tooth",
            "impacted_analysis": impacted_analysis
        }
