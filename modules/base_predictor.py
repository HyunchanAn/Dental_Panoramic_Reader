from abc import ABC, abstractmethod
import numpy as np

class BasePanoramicPredictor(ABC):
    @abstractmethod
    def load_model(self, *args, **kwargs) -> None:
        """모델 로드 및 초기화"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> dict:
        """
        단일 파노라마 이미지에 대한 추론 수행
        반환값은 지정된 표준 데이터 구조(Schema)를 준수해야 함
        """
        pass
