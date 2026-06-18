import traceback
import logging

class PredictorRegistry:
    def __init__(self):
        self._predictors = {}

    def register_module(self, name: str, predictor):
        self._predictors[name] = predictor

    def get_available_modules(self):
        return list(self._predictors.keys())

    def run_pipeline(self, image, selected_modules: list, **kwargs):
        """선택된 모듈들을 순차/병렬로 구동하고 결과를 취합"""
        results = {}
        for name in selected_modules:
            if name in self._predictors:
                try:
                    # 각 예측기에 특화된 kwargs가 있을 수 있으므로 모두 전달
                    res = self._predictors[name].predict(image, **kwargs)
                    res["status"] = "success"
                    results[name] = res
                except Exception as e:
                    logging.error(f"Error running module {name}: {traceback.format_exc()}")
                    results[name] = {
                        "module_name": name,
                        "status": "error",
                        "error_message": str(e)
                    }
        return results
