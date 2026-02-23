# JiT 1채널(Grayscale) 변환 및 최적화 가이드

본 프로젝트는 원본 JiT 모델을 Windows 환경, 단일 GPU, 1채널(Grayscale) 이미지 데이터셋에 맞게 최적화하고 가중치를 이식한 버전입니다.

---

## 1. 공통 수정 사항 (학습 및 추론 공통)

- **모델 입력 채널 수정 (`model_jit.py`)**
  - `BottleneckPatchEmbed` 클래스의 `in_chans`를 3에서 1로 변경하여 흑백 이미지 입력 대응.

- **모델 출력 채널 수정 (`model_jit.py`)**
  - `FinalLayer` 클래스의 최종 출력 차원을 흑백 픽셀 수($PatchSize^2 \times 1$)에 맞게 조정.

- **노이즈 생성 로직 수정 (`denoiser.py`)**
  - 가우시안 노이즈 생성 시 채널 크기를 1로 고정하여 1채널 텐서 흐름 유지.

- **사전학습 가중치 채널 보정 (`main_jit.py`)**
  - 입력(`proj1`): 3채널(RGB) 가중치의 평균(Mean)을 내어 1채널 구조로 변환 로드.
  - 출력(`final_layer`): 3채널 분량의 출력 가중치를 평균 처리하여 1채널로 이식.

- **Triton 의존성 제거**
  - Windows 환경 호환성을 위해 `@torch.compile` 데코레이터 및 Triton 관련 로직 제거.

---

## 2. 학습(Train) 전용 수정 사항

- **데이터 로딩 파이프라인 수정 (`main_jit.py`)**
  - `Image.open().convert('L')`를 적용하여 모든 학습 데이터를 8-bit Grayscale로 강제 변환.

- **정규화 파라미터 변경 (`main_jit.py`)**
  - 3채널용 `mean`, `std`를 흑백용 단일 값(예: `[0.5]`)으로 수정하여 데이터 분포 최적화.

- **해상도 및 패치 사이즈 정합성**
  - 512 해상도 학습 시 패치 사이즈를 32로 설정하여, 256 해상도(패치 16) 사전학습 모델의 위치 임베딩(`pos_embed`) 개수와 일치시킴.

- **단일 GPU 학습 지원**
  - 분산 학습 관련 `DDP` 및 `DistributedSampler`를 제거하고 `RandomSampler`를 통한 단일 장치 학습 지원.

---

## 3. 추론(Inference) 전용 수정 사항

- **결과 이미지 가공 및 저장 (`engine_jit.py`)**
  - 모델 출력 텐서의 채널 차원 제거를 위해 `squeeze()` 처리.
  - 출력 범위 $[-1, 1]$을 $[0, 255]$로 복원하는 정규화 해제 수식 적용: `(x + 1) / 2 * 255`.
  - `cv2.imwrite` 시 추가 변환 없이 단일 채널 Grayscale 파일로 저장.

- **CFG(Classifier-Free Guidance) 강화**
  - 1채널 변환 후 약해진 신호와 윤곽선을 보강하기 위해 `cfg` 값을 기존보다 높게(7.0~10.0) 설정 권장.

- **클래스 임베딩 유연 로드 (`strict=False`)**
  - 체크포인트의 클래스 개수가 현재 설정과 다를 경우, 가중치를 슬라이싱하여 필요한 만큼만 로드.

---

## 실행 방법

### 추론 (Inference)

```bash
python main_jit.py --config inference.json
```

### 학습 (Train)

```bash
python main_jit.py --data_path [데이터경로] --model JiT-B/32 --img_size 512
```
