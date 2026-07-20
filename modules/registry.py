import torch
import gc
import traceback
import logging
import concurrent.futures

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
        base_modules = ["Dental_008_segmentation"]
        
        # Stage 1: Run Base Modules
        for name in selected_modules:
            if name in self._predictors and name in base_modules:
                try:
                    res = self._predictors[name].predict(image, **kwargs)
                    res["status"] = "success"
                    results[name] = res
                    if hasattr(self._predictors[name], "unload_model"):
                        self._predictors[name].unload_model()
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error running base module {name}: {traceback.format_exc()}")
                    results[name] = {"module_name": name, "status": "error", "error_message": str(e)}

        # 의존성 주입: 008의 결과가 성공적이라면 추출하여 kwargs에 추가
        if "Dental_008_segmentation" in results and results["Dental_008_segmentation"]["status"] == "success":
            teeth_data = results["Dental_008_segmentation"].get("teeth", [])
            kwargs["teeth_data"] = teeth_data
            kwargs["fdi_list"] = [t["fdi"] for t in teeth_data if "fdi" in t]

        # Stage 2: Run Derived Modules (Those requiring 008 outputs)
        derived_modules = [name for name in selected_modules if name in self._predictors and name not in base_modules]
        
        def run_derived(name):
            try:
                res = self._predictors[name].predict(image, **kwargs)
                res["status"] = "success"
                if hasattr(self._predictors[name], "unload_model"):
                    self._predictors[name].unload_model()
                return name, res
            except Exception as e:
                logging.error(f"Error running derived module {name}: {traceback.format_exc()}")
                return name, {"module_name": name, "status": "error", "error_message": str(e)}

        # VRAM 한계(8GB)를 고려하여 Max Workers를 2~3개로 제한합니다.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_name = {executor.submit(run_derived, name): name for name in derived_modules}
            for future in concurrent.futures.as_completed(future_to_name):
                name, res = future.result()
                results[name] = res
                torch.cuda.empty_cache()
                gc.collect()

        return results
