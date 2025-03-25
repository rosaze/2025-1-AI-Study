# Attention Is All You Need

## 🔹 기존 RNN, GRU 기반 언어 모델의 한계

- **순차적 계산의 제약**  
  - 문자의 위치에 따라 계산이 순차적으로 진행됨  
  - 병렬 처리 불가 → 여러 단어를 동시에 처리하지 못함  
- **메모리 한계**  
  - 긴 문장을 처리할 경우, 여러 문장을 한 번에 학습하기 어려움  

## 🔹 Attention 메커니즘이란?

- Input과 Output 시퀀스 간 거리와 무관하게 단어 간의 의존성을 학습  
- 기존에는 대부분 RNN과 함께 사용됨

## 🔹 CNN 기반 모델의 특징과 한계

- Input과 Output을 동시에 계산 → **병렬 처리 가능**  
- 하지만 input, output 간 관계 연결 시 연산량 급증

## 🔹 Self-Attention의 핵심

- 시퀀스 내 서로 다른 위치 간의 관계를 학습하여 전체 표현을 한 번에 계산 
- 장거리 관계도 한 단계로 연결 가능  
- 전체 시퀀스를 병렬 처리 → 빠른 학습 가능  
- **연산 효율성 비교**:
  - **RNN**: 순차적 처리 → 연산량 O(n)
  - **CNN**: 병렬화 가능하지만 여러 층 필요 → 비용 증가
  - **Self-Attention**: 연산량 O(n²·d), 병렬화 가능 → 대부분의 실제 상황에서 더 빠름

---

## 🔹 Transformer 구조: Encoder, Decoder

- 기존 RNN 기반 방식은 Input, Output을 순서대로 생성  
- Transformer는 **Self-Attention + Fully Connected Layer** 조합을 반복적으로 사용

---

### 🔹 Encoder: 입력 시퀀스를 연속적인 벡터 시퀀스로 변환

- 총 **6개의 레이어**로 구성
- 각 레이어는 아래 두 가지 주요 구성 요소로 구성됨:
  - **Multi-Head Self-Attention**: 문장 내 단어들의 상호 관계를 파악
  - **Position-wise Feed Forward Network**: 각 단어 위치에서 독립적인 계산 수행

- 각 sub-layer에는 다음이 포함됨:
  - **잔차 연결 (Residual Connection)**:  
    레이어의 input에 output을 더하여 출력 → 원래 정보를 유지하며 학습 가능
  - **레이어 정규화 (Layer Normalization)**:  
    데이터의 분포를 고르게 조정하여 학습을 안정화  
    (→ 각 벡터 내부 값을 평균 0, 분산 1로 정규화)

- 전체 모델 및 임베딩 출력 차원: **512차원**

---

### 🔹 Decoder: 인코더의 출력을 바탕으로 출력 시퀀스를 순차적으로 생성

- 총 **6개의 레이어**로 구성
- 주요 구성 요소:
  - **Masked Multi-Head Self-Attention**:  
    미래 단어 정보를 보지 않도록 **masking** → 과거 출력만을 참조하여 다음 단어 생성
  - **Multi-Head Attention over Encoder Output**:  
    인코더에서 생성된 벡터를 참고하여 더 정확한 출력 생성
  - **Position-wise Feed Forward Network**:  
    각 단어 위치에서 추가적인 계산 수행

## 🔹 Attention이란?

입력 시퀀스의 단어들 간 관계를 계산해주는 메커니즘

### 🔹 Input 구성 요소
- **Query**: 기준이 되는 단어  
- **Key**: 비교 대상이 되는 단어들  
- **Value**: 실제 정보를 담고 있는 값  
- → Query와 Key의 **유사도**를 계산한 후, 그 유사도를 바탕으로 Value를 가중합해 출력 생성

---

## 🔹 Output 계산 방식: Scaled Dot-Product Attention

- Query와 Key의 유사도를 **Dot Product**로 계산  
- → 유사도 점수가 커질수록 Softmax를 통과할 때 기울기가 0에 가까워질 수 있음 → 학습 불안정

- 따라서 **√dk (dk는 query/key의 차원 수)** 로 나누어 **스케일링** → 수치 안정화
- Softmax를 통해 점수를 **확률 분포**로 변환  
  - Softmax는 여러 점수를 입력받아, 각 값을 0~1 사이의 확률로 바꾸고  총합이 1이 되도록 정규화해주는 함수  
- 각 Value에 해당 확률을 곱하고, 그 합을 최종 출력으로 사용
- 이 연산은 **행렬곱(Matrix Multiplication)**으로 구현되어  
  속도와 메모리 효율이 매우 뛰어남

---

## 🔹 Multi-Head Attention이란?

attention을 **여러 번 병렬로 수행**하는 방식

- 서로 다른 관점에서 입력 시퀀스를 해석할 수 있도록  
  여러 개의 attention head를 동시에 수행  
- 각 head는 독립적으로 query, key, value를 생성  
- 다양한 위치 간의 관계를 병렬로 학습  
→ 속도 빠르고, 모델이 **다양한 위치에 동시에 집중(attend)** 가능

---

## 🔹 Transformer에서 사용되는 3가지 Multi-Head Attention 방식

### 🔹 Encoder-Decoder Attention
- Query: Decoder의 이전 출력  
- Key & Value: Encoder의 출력  
- → Decoder의 각 위치가 **입력 전체에 주의(attend)** 함

### 🔹 Encoder Self-Attention
- Query, Key, Value 모두 Encoder 내부의 같은 입력에서 생성  
- → 인코더의 각 단어가 **다른 모든 단어 위치에 attend** 가능

### 🔹 Decoder Self-Attention
- Query, Key, Value 모두 Decoder 내부에서 생성  
- → 미래 단어를 보지 않도록 **마스킹 처리**  
  - 마스킹: 미래 위치에 대한 유사도 점수를 -∞로 설정하여 Softmax 결과를 0으로 만듦  
  - → **현재 위치까지의 정보만 활용** 가능하게 함

## 🔹 Fully Connected Feed-Forward Network (FNN)

- Transformer의 각 레이어에서 **모든 위치에 독립적으로 적용**
- 구조: **2개의 선형 변환 + ReLU**  
  - **ReLU(Rectified Linear Unit)**:  
    음수는 0으로 만들고 양수는 그대로 통과시키는 비선형 함수 → 연산 간단하고, 기울기 소실 문제 완화
  → 복잡한 비선형 관계 학습 가능

---

## 🔹 Embedding

- Transformer는 입력 토큰과 출력 토큰을 **d_model 차원의 벡터로 변환**  
  → **Input Embedding**, **Output Embedding**, **Softmax 이전 선형 계층**에 **같은 가중치 행렬**을 공유  
  → 파라미터 수를 줄이고 일반화 성능을 향상시킴
- 최종 출력 생성 시 **Linear Transformation + Softmax** 사용

---

## 🔹 Positional Encoding: 입력 토큰의 순서를 인식하는 방법

- RNN이나 CNN이 없기 때문에 **위치 정보를 직접 주입해야 함**
- 각 임베딩 벡터에 **위치 벡터를 더함**
- 위치 벡터는 각 차원마다 **사인/코사인 함수**를 적용하여 구성  
  → 긴 시퀀스에도 잘 일반화됨  
  → 주파수는 기하급수적으로 증가 → 다양한 위치 차이를 인식 가능

---

## 🔹 Training (학습 방식)

- Transformer는 **대규모 번역 데이터**로 학습됨
- **Cross Entropy Loss + Adam Optimizer + Learning Rate Scheduling** 사용

  - **Cross Entropy Loss**:  
    모델이 예측한 확률 분포와 실제 정답 간의 차이를 계산  
    → 정답 토큰에 가까운 확률을 높게 예측할수록 손실 값이 낮아짐  
    → 분류 문제에서 가장 많이 사용되는 손실 함수

  - **Adam Optimizer**:  
    파라미터를 어떻게 업데이트할지 결정 → 모멘텀 + 적응적 학습률을 결합하여  
    빠르고 안정적인 학습 가능

  - **Learning Rate Scheduling**:  
    - 초기: **Warm-up** → 학습률을 점차 증가  
    - 이후: **Decay** → 점차 감소하며 정밀하게 학습

  - **Label Smoothing**:  
    정답 토큰의 확률을 100%로 고정하지 않고 **약간 퍼뜨려서 학습**  
    → 예: 정답 클래스에 0.9, 나머지에 소량 분배  
    → 모델이 **너무 확신하게 되는 것(overconfidence)**을 막아 **일반화 성능 향상** (과적합방지지)

---

## 🔹 Model Variations (모델 성능에 영향을 주는 구성 요소)

- **Attention Head 수**  
  → 너무 적거나 많으면 성능 저하
- **Key/Value 벡터 차원 수**  
  → 너무 작으면 성능 저하
- **FNN 차원 크기**  
  → 클수록 성능 증가
- **Dropout**  
  → 과적합 방지에 효과적
- **Learned Positional Embeddings vs. Sinusoidal Encoding**  
  → 결과 성능은 유사