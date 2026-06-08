import os
import json
import shutil
from pathlib import Path

# 경로 설정
BASE_DIR = Path(r"c:\Users\chema\Github\Caries_Detection_from_Panoramic")
RAW_DATA_DIR = BASE_DIR / "data/Dental_dataset/Pediatric dental disease detection dataset/Train"
OUTPUT_DIR = BASE_DIR / "data/processed/train"  # 기존 학습 데이터 위치에 병합

# 클래스 매핑 (중국어 -> YOLO Class ID/Name)
# data.yaml의 클래스 순서: 0: Impacted, 1: Caries, 2: Periapical Lesion, 3: Deep Caries
LABEL_MAP = {
    "龋病": 1,          # Caries
    "深窝沟": 3,        # Deep Caries
    "根尖周炎": 2,      # Periapical Lesion
    "牙齿发育异常": 0,  # Impacted (발육 이상을 매복으로 처리)
    "牙髓炎": 3         # Deep Caries (치수염을 심한 충치로 통합)
}

def convert():
    print("소아 데이터셋 변환 시작...")
    
    # 출력 폴더 생성
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    json_files = list((RAW_DATA_DIR / "label").glob("*.json"))
    print(f"총 {len(json_files)}개의 라벨 파일을 발견했습니다.")

    count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보
            img_h = data.get('imageHeight')
            img_w = data.get('imageWidth')
            if not img_h or not img_w:
                print(f"Skipping {json_file.name}: 이미지 크기 정보 없음")
                continue

            # 라벨 변환
            yolo_lines = []
            for shape in data.get('shapes', []):
                label_cn = shape.get('label')
                class_id = LABEL_MAP.get(label_cn)
                
                if class_id is None:
                    continue # 매핑되지 않은 라벨 무시

                # 좌표 변환
                points = shape.get('points')
                if not points or len(points) < 2:
                    continue
                
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # 좌표 클램핑 (이미지 범위 내로)
                x_min = max(0, min(img_w, x_min))
                x_max = max(0, min(img_w, x_max))
                y_min = max(0, min(img_h, y_min))
                y_max = max(0, min(img_h, y_max))

                # YOLO Center format normalization
                dw = 1.0 / img_w
                dh = 1.0 / img_h
                
                w = x_max - x_min
                h = y_max - y_min
                x_center = x_min + w / 2.0
                y_center = y_min + h / 2.0

                x_center *= dw
                w *= dw
                y_center *= dh
                h *= dh

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 저장 (파일명 충돌 방지를 위해 접두어 추가)
            file_stem = json_file.stem
            new_filename = f"ped_{file_stem}"
            
            if yolo_lines: 
                # 1. 라벨 파일 저장
                with open(OUTPUT_DIR / "labels" / f"{new_filename}.txt", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))

                # 2. 이미지 파일 복사
                src_img = RAW_DATA_DIR / "images" / f"{file_stem}.png" 
                # 일부 파일이 jpg일 수도 있으니 확인 (현재 데이터셋은 주로 png로 보임)
                if not src_img.exists():
                     src_img = RAW_DATA_DIR / "images" / f"{file_stem}.jpg"

                if src_img.exists():
                    dst_img = OUTPUT_DIR / "images" / f"{new_filename}{src_img.suffix}"
                    shutil.copy2(src_img, dst_img)
                    count += 1
                else:
                    print(f"이미지 파일을 찾을 수 없음: {file_stem}")

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    print(f"변환 및 복사 완료: {count}개 파일 처리됨.")

if __name__ == "__main__":
    convert()
