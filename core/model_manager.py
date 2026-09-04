import gc
import torch

class ModelManager:
    """
    RTX 4060 Laptop (8GB VRAM) 환경에서 다수의 AI 모델을 구동하기 위한 VRAM 최적화 매니저.
    사용하지 않는 모델을 CPU로 오프로딩하거나 캐시를 비워 OOM을 방지합니다.
    PyTorch (.pt)뿐만 아니라 ONNX (.onnx) 래퍼 객체도 예외 없이 안전하게 등록 및 관리합니다.
    """
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def register_model(self, name, model):
        """모델을 등록하고 PyTorch 가중치인 경우에만 CPU에 둡니다 (ONNX 래퍼는 .to() 호출 제외)."""
        if model is not None and hasattr(model, 'to'):
            # Check if it is an ONNX wrapper or non-PyTorch YOLO model
            is_onnx = getattr(model, 'is_patch_model', False) or (
                hasattr(model, 'model_name') and str(model.model_name).endswith('.onnx')
            ) or (
                hasattr(model, 'overrides') and str(model.overrides.get('model', '')).endswith('.onnx')
            )
            if not is_onnx:
                try:
                    model = model.to('cpu')
                except (TypeError, AttributeError):
                    pass
        self.models[name] = model
    
    def get_model(self, name):
        return self.models.get(name)

    def load_to_gpu(self, name):
        """특정 모델을 GPU로 올립니다. 이때 다른 PyTorch 모델들은 CPU로 내립니다."""
        for n, m in self.models.items():
            if m is not None and hasattr(m, 'to'):
                is_onnx = getattr(m, 'is_patch_model', False) or (
                    hasattr(m, 'model_name') and str(m.model_name).endswith('.onnx')
                ) or (
                    hasattr(m, 'overrides') and str(m.overrides.get('model', '')).endswith('.onnx')
                )
                if not is_onnx:
                    try:
                        if n == name:
                            m.to(self.device)
                        else:
                            m.to('cpu')
                    except (TypeError, AttributeError):
                        pass
        
        # PyTorch VRAM 캐시 강제 비우기
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return self.models.get(name)
    
    def clear_cache(self):
        """사용 가능한 모든 GPU VRAM을 강제로 확보합니다."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
