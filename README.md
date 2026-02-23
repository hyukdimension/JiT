# ORIG 대비 수정 사항

## 핵심 목적

**Linux 분산 학습 → Windows 단일 GPU, Tiny ImageNet**으로 전환, Triton 비사용

## main_jit.py

1. **인퍼런스 결과 이미지들을 스텝별로 폴더 만들어 모두 저장**
   - 배경: 스텝 진행에 따른 결과 이미지의 변화를 보기위해.

2. **Config JSON 파일 로드 기능 추가**
   - `--config` 인자로 설정 파일 지원
   - 배경: 사내 관례

3-1. **Windows 멀티프로세싱 호환 함수 추가**
   - `setup_runtime_env()` 및 분산 학습 함수 더미 구현
   - 배경: 싱글 gpu 환경 체택에 따라 분산처리 관련부 모두 비활성화


3-2. **분산 학습 코드 제거**
   - `DistributedDataParallel`, `DistributedSampler` 제거
   - `RandomSampler`로 변경 (단일 GPU)
   - 배경: 싱글 gpu 환경 체택에 따라 분산처리 관련부 모두 비활성화

4. **TinyImageNet 커스텀 데이터셋 클래스 추가**
   - ImageNet 대신 Tiny ImageNet 지원
   - 테스트 기간 동안 Tiny ImageNet 데이터셋 활용에 따라.

5. **체크포인트 로드 로직 개선**
   - 파일/폴더 모두 처리

## model_jit.py

1. **`@torch.compile` 데코레이터 제거** (FinalLayer, JiTBlock)
   - Windows Triton 컴파일 에러 방지
   - 배경: 윈도우에서는 Triton이 지원 안됨

2. **autocast 구문 변경**
   - `torch.cuda.amp.autocast(enabled=False)` → `torch.amp.autocast('cuda', dtype=torch.bfloat16)`
   - 배경: Deprecated 표현임. 그리고 인자가 바뀌어서, 첫번째 인자에 device 명을 적어주어야 함

---

## 사용법

### 학습
```cmd
activate.bat
train.bat train.json
```

### 추론
```cmd
activate.bat
inference.bat inference.json
```
