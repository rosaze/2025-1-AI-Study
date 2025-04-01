# 📄 논문 리뷰: Attention Is All You Need

---

## 핵심 요약

- 기존 RNN/CNN 기반 시퀀스 모델의 한계를 극복하기 위해 **Transformer** 아키텍처 제안
- **Self-Attention**만 사용해 병렬 처리 가능, 장거리 의존성 학습 용이
- **WMT 2014 영어→독일어 BLEU 28.4, 영어→프랑스어 BLEU 41.8**로 기존 SOTA 모델보다 뛰어난 성능
- 구문 분석(Parsing) 등 다른 NLP 과제에도 높은 일반화 성능

---

## 주요 내용 요약

### 모델 구조
- **인코더-디코더 구조** 유지, RNN 제거
- 모든 층은 `Multi-Head Self-Attention + Feed Forward + LayerNorm + Residual`
- **Positional Encoding**으로 단어 순서 정보 전달

### Attention 메커니즘
- **Scaled Dot-Product Attention**: Q, K, V 기반 계산
- **Multi-Head Attention**: 다양한 표현 공간에서 정보 추출

### 학습 설정
- Optimizer: Adam + 커스텀 learning rate 스케줄
- 정규화: Dropout (0.1~0.3), Label Smoothing (ε=0.1)
- 데이터: WMT14 EN-DE (4.5M), EN-FR (36M), BPE 사용

---

## 성능 요약

| 모델               | EN-DE BLEU | EN-FR BLEU |
|--------------------|------------|------------|
| Transformer (base) | 27.3       | 38.1       |
| Transformer (big)  | **28.4**   | **41.8**   |

- 학습 시간: base 모델 12시간, big 모델 3.5일 (8 × P100 GPU)
- 기존 모델 대비 **짧은 학습 시간 + 높은 성능**

---

## 관찰 결과

- 어텐션 헤드 수 증가 → 다양한 정보 처리 가능
- positional encoding은 학습식/사인코사인식 모두 유사 성능
- 레이블 스무딩과 Dropout은 BLEU 향상에 효과적
- 기계 번역 외에도 WSJ 구문 분석에서 F1 = 92.7 기록

---

## 결론

Transformer는 RNN/CNN 없이 **어텐션만으로 시퀀스 변환이 가능**함을 증명
이후 등장하는 BERT, GPT, T5 등 모든 대형 NLP 모델의 **기초가 되는 아키텍처**

---
