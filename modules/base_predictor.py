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
        
    def to(self, device):
        """
        래퍼 내부의 실제 모델(self.model) 객체를 주어진 디바이스(cpu/cuda)로 전송합니다.
        PyTorch나 Ultralytics 원시 모델의 .to() 인터페이스를 흉내내어 ModelManager와 호환되게 합니다.
        """
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'to'):
                self.model.to(device)
        return self
