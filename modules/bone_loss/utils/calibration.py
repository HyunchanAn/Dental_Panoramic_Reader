class CalibrationManager:
    """
    파노라마 이미지의 픽셀 단위를 물리적 거리(mm)로 매핑하는 유틸리티 클래스입니다.
    """
    
    def __init__(self, pixels_per_mm: float = 1.0):
        """
        초기 스케일 팩터를 설정합니다. 기본값은 1.0 (Pixel = mm) 입니다.
        """
        self.pixels_per_mm = pixels_per_mm

    def set_scale_from_dicom(self, pixel_spacing: list):
        """
        DICOM 파일의 PixelSpacing 태그 값을 기반으로 스케일을 설정합니다.
        보통 [row_spacing, col_spacing] 형태를 가지며 평균을 사용합니다.
        """
        if pixel_spacing and len(pixel_spacing) >= 2:
            avg_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
            if avg_spacing > 0:
                self.pixels_per_mm = 1.0 / avg_spacing

    def set_scale_from_reference(self, reference_pixel_length: float, reference_real_length_mm: float):
        """
        이미지 내 기지 구조물(예: 임플란트 픽스처, Calibration Ball)의 
        픽셀 길이와 실제 길이를 바탕으로 스케일을 산출합니다.
        """
        if reference_pixel_length > 0 and reference_real_length_mm > 0:
            self.pixels_per_mm = reference_pixel_length / reference_real_length_mm
            
    def pixel_to_mm(self, pixel_distance: float) -> float:
        """
        주어진 픽셀 거리를 물리적 mm 거리로 변환합니다.
        """
        return pixel_distance / self.pixels_per_mm
