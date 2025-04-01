# 📄 논문 리뷰: Zero-Shot Text-to-Image Generation

**링크**: [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)

---

## 연구 목적
텍스트 설명으로부터 이미지를 생성하는 모델을 개발. 별도의 도메인 특화 학습 없이도 다양한 이미지를 생성할 수 있는 **zero-shot** 성능을 목표로 함.

---

## 모델 구조

### Stage 1: dVAE (Discrete VAE)
- 이미지를 32x32 토큰으로 압축 (총 1024개)
- 각 토큰은 8192개 코드북 벡터 중 하나
- 이미지 정보를 효율적으로 Transformer가 처리할 수 있게 변환

### Stage 2: Transformer
- 텍스트 토큰 (최대 256개) + 이미지 토큰 (1024개)을 하나의 시퀀스로 연결
- Autoregressive 방식으로 joint distribution 모델링
- 모델 크기: 12B 파라미터, 64층 Transformer

---

## 실험 및 성능

### 데이터셋
- 2.5억 개의 텍스트-이미지 쌍 수집 (Conceptual Captions, YFCC100M 등)

### 평가 결과
- MS-COCO 캡션에 대해 zero-shot으로 DF-GAN 등 기존 모델보다 높은 선호도
- FID, IS 점수에서도 blur radius에 따라 가장 우수한 성능

---

## 특이사항
- "a tapir made of accordion" 같은 창의적인 문장에도 적절한 이미지 생성
- 텍스트 렌더링, 스타일 전환, 간단한 이미지 변환까지 가능
- 단일 모델로 다양한 작업이 가능하다는 점에서 확장성 뛰어남

---

## 결론
> "텍스트를 입력하면 이미지를 그려주는 모델을, 별도 학습 없이도 잘 작동하게 만든 대규모 Transformer 기반 접근"

---
