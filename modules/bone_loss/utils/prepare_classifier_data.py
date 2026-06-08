import os
import shutil
import urllib.request
import glob
import time

def prepare_data():
    base_dir = "data/classifier"
    pano_dir = os.path.join(base_dir, "pano")
    non_pano_dir = os.path.join(base_dir, "non_pano")
    
    os.makedirs(pano_dir, exist_ok=True)
    os.makedirs(non_pano_dir, exist_ok=True)
    
    print("1. 파노라마(Positive) 데이터 복사 중...")
    pano_source = "ufba-425-1/train/images/*.jpg"
    pano_files = glob.glob(pano_source)
    
    if not pano_files:
        print("경고: ufba-425-1 데이터셋을 찾을 수 없습니다.")
    else:
        # 학습 속도를 위해 200장 정도만 샘플링하여 복사
        for i, filepath in enumerate(pano_files[:200]):
            shutil.copy(filepath, os.path.join(pano_dir, f"pano_{i}.jpg"))
        print(f"-> {len(pano_files[:200])}장의 파노라마 사진 복사 완료.")
        
    print("2. 일반 OOD(Negative) 기본 데이터 다운로드 중...")
    # Picsum API를 활용해 일반적인 랜덤 사진 100장을 다운로드하여 non_pano 폴더에 저장
    # (풍경, 사물 등 파노라마가 아닌 완벽한 OOD)
    try:
        for i in range(100):
            url = f"https://picsum.photos/400/400?random={i}"
            urllib.request.urlretrieve(url, os.path.join(non_pano_dir, f"random_ood_{i}.jpg"))
            if i % 20 == 0:
                print(f"-> {i}/100 다운로드 중...")
            time.sleep(0.1)
        print("-> 100장의 랜덤 일반 사진 다운로드 완료.")
    except Exception as e:
        print(f"다운로드 중 오류 발생: {e}")
        
    print("\n" + "="*60)
    print("✅ 기초 데이터 세팅이 완료되었습니다!")
    print(f"- 파노라마 정답 폴더: {os.path.abspath(pano_dir)}")
    print(f"- 오답(OOD) 폴더: {os.path.abspath(non_pano_dir)}")
    print("="*60)
    print("\n[사용자님께 드리는 필수 요청 사항]")
    print("지금 오답 폴더에는 강아지나 풍경 같은 완전한 일반 사진만 들어있습니다.")
    print("진짜 똑똑한 컷(Cut) 필터를 만들려면, 치근단 방사선 사진(Periapical X-ray)이나")
    print("세팔로(Cephalo) 같은 '엑스레이지만 파노라마는 아닌' 사진들이 필요합니다.")
    print(f"👉 구글 검색 등을 통해 치근단 방사선 사진 이미지를 10장~20장 정도만 캡처해서")
    print(f"   위의 [오답(OOD) 폴더] 경로 안에 파일명 상관없이 넣어주세요!")
    print("작업이 끝나면 에이전트에게 '사진 넣었어, 학습 시작해' 라고 말씀해 주시면 됩니다.")

if __name__ == "__main__":
    prepare_data()
