# 2.4 Mixed-Precision Training

🔹 **문제 상황**  
- AI 모델이 커질수록 GPU 메모리에 한계 발생  
- 32비트 연산 → 16비트로 바꾸면 메모리는 절반, 속도는 빨라짐  
- 하지만 16비트는 너무 작은 숫자를 표현 못해서  
  → 값이 0처럼 되어버리는 **underflow** 현상 발생  
- 특히 앞쪽 → 뒤쪽 ResBlock으로 갈수록  
  → `gradient norm`(기울기의 크기)이 점점 작아짐

🔹 **해결책**  
- 각 블록마다 gradient 크기를 따로 조절  
  → `per-resblock gradient scaling`

🔹 **압축률**  
- 약 **85%까지 메모리 절약** 가능

🔹 **기존 해결법: Loss Scaling**  
- gradient 값의 범위가 16비트 표현 범위 내에 있어야 효과적  
- 텍스트-이미지 생성 모델에선 범위가 너무 작아서 효과 제한적

🔹 **새로운 해결책: ResBlock별 Gradient Scale**  
- Flexpoint라는 고급 혼합 정밀도 학습 기법의 현실적 대안  
- 특수 GPU 장비나 커널 없이도 구현 가능

🔹 **이때 Flexpoint란?**  
- 딥러닝 학습 중 숫자를 더 가볍고 안정적으로 표현하기 위한 포맷  
- 정밀도(precision)와 표현 범위(exponent range)를 동시에 조절  
→ 연산 효율 + 메모리 절약 + 안정성

---

# 2.5 Distributed Optimization (분산 최적화)

🔹 문제 상황  
- 모델 파라미터: 120억 개  
- 16비트로 저장해도 약 24GB 메모리 필요   
- 하지만 GPU 한 개는 보통 16GB → 초과됨

🔹 해결책 1: **Parameter Sharding**  
- 여러 GPU가 모델 파라미터를 나눠 저장하고 계산  
- 같은 컴퓨터 안의 GPU끼리는 통신 빠름  
- 하지만 서로 다른 컴퓨터끼리는 느림 → 병목 발생

🔹 해결책 2: **PowerSGD (Gradient 압축)**  
- gradient를 압축해서 GPU 간 통신량을 줄이는 기술

🔹 작동 방식  
- 각 GPU는 자기 gradient를 low-rank로 압축  
- 오차는 아래처럼 처리:  
  1. 8개 GPU가 만든 gradient 평균  
  2. 압축한 gradient를 다시 복원한 값  
  3. 이 차이를 `error buffer`에 저장  
  4. 이후 학습에 누적해 정밀도 유지

🔹 통신 효율 
- 기존 방식: 압축되지 않은 큰 gradient 전체 전송  
- PowerSGD: 작은 2개의 low-rank 요소만 전송  
→ 속도 빠름, 메모리 절약

🔹 압축률 공식  
- r: 압축 랭크  
- d_model: 트랜스포머 내부 차원  
→ 약 **85% 압축 가능**

🔹 세부 최적화 사항  
- gradient를 별도 버퍼 없이 error buffer에 바로 누적해 메모리 절약  
- NaN 발생이나 재시작 외엔 error buffer를 지우지 않음  
- Gram-Schmidt 대신 Householder 정규화를 사용하여 수치 안정성 향상  
- 입력에 작은 항등행렬 추가  
- 언더플로 방지를 위해 커스텀 16비트 실수 포맷 사용
