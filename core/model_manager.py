import gc
import torch

class ModelManager:
    """
    RTX 4060 Laptop (8GB VRAM) 환경에서 다수의 AI 모델을 구동하기 위한 VRAM 최적화 매니저.
    사용하지 않는 모델을 CPU로 오프로딩하거나 캐시를 비워 OOM을 방지합니다.
    """
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def register_model(self, name, model):
        """모델을 등록하고 기본적으로 CPU에 둡니다."""
        if hasattr(model, 'to'):
            model = model.to('cpu')
        self.models[name] = model
    
    def get_model(self, name):
        return self.models.get(name)

    def load_to_gpu(self, name):
        """특정 모델을 GPU로 올립니다. 이때 다른 모델들은 CPU로 내립니다."""
        for n, m in self.models.items():
            if hasattr(m, 'to'):
                if n == name:
                    m.to(self.device)
                else:
                    m.to('cpu')
        
        # PyTorch VRAM 캐시 완전 삭제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return self.models.get(name)
    
    def clear_cache(self):
        """사용 가능한 모든 GPU VRAM을 강제로 확보합니다."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
