# 문장 내 개체간 관계 추출

[TOC]

## Hardware

- `GPU : Tesla V100 32GB`

## Project Description

> 문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다. 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.
> 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.
> 이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 model이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```

## 평가 방법

KLUE-RE evaluation metric을 그대로 재현했습니다.

1. no_relation class를 제외한 **micro F1 score**
2. 모든 class에 대한 **area under the precision-recall curve (AUPRC)**

- 2가지 metric으로 평가하며, **micro F1 score**가 우선시 됩니다.

Micro F1 score

- micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여합니다. 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산 됩니다.

AUPRC

- x축은 Recall, y축은 Precision이며, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정 합니다. imbalance한 데이터에 유용한 metric 입니다

## Dataset

- dataset 설명

```
data
|    +- train_pororo_sub.csv
|    +- test_pororo_sub.csv
|    +- train.csv
|    +- test.csv
```

    - 'train_pororo_sub.csv'를 활용하여 `RBERT`, `KLUE/RoBERTa-large` 학습을 진행한다.
    - 'test_pororo_sub.csv'를 활용하여 `RBERT`, `KLUE/RoBERTa-large` 모델을 바탕으로 'submission.csv' 파일을 생성한다.
    - 'train.csv'를 활용하여 `RE Improved Baseline` 학습을 진행한다.
    - 'test.csv'를 활용하여 `RE Improved Baseline` 모델을 바탕으로 'submission.csv' 파일을 생성한다.

- Dataset 통계
  - train dataset : 총 32470개
  - test dataset : 7765개 (label은 전부 100으로 처리되어 있습니다.)
- Data 예시 (`train.csv`)
  - `id`, `sentence`, `subject_entity`, `object_entity`, `label`, `source`로 구성된 csv 파일
  - `sentence example` : <Something>는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다. (문장)
  - `subject_entity example` : {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'} (단어, 시작 idx, 끝 idx, 타입)
  - `object_entity example` : {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} (단어, 시작 idx, 끝 idx, 타입)
  - `label example` : no_relation (관계),
  - `source example` : wikipedia (출처)
- Relation class에 대한 정보는 다음과 같습니다.
  ![1](https://user-images.githubusercontent.com/53552847/136692171-30942eec-fb83-4175-aa8d-13559ae2caf1.PNG)

## code

- `train.py`

  - code를 학습시키기 위한 파일입니다.
  - 저장된 model관련 파일은 `results` 폴더에 있습니다.

- `inference.py`

  - 학습된 model을 통해 prediction하며, 예측한 결과를 csv 파일로 저장해줍니다.
  - 저장된 파일은 prediction 폴더에 있습니다.

- `load_data.py`

  - baseline code의 전처리와 데이터셋 구성을 위한 함수들이 있는 코드입니다.

- `logs`

  - 텐서보드 로그가 담기는 폴더 입니다.

- `prediction`

  - `inference.py` 를 통해 model이 예측한 정답 `submission.csv` 파일이 저장되는 폴더 입니다.

- `results`

  - `train.py`를 통해 설정된 step 마다 model이 저장되는 폴더 입니다.

- `best_model `

  - 학습중 evaluation이 best인 model이 저장 됩니다.

- `dict_label_to_num.pkl`

  - 문자로 되어 있는 label을 숫자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

- `dict_num_to_label.pkl`
  - 숫자로 되어 있는 label을 원본 문자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

## Implementation

In Terminal

- Install Requirements

```python
pip install -r requirements.txt
```

- training

```
python train.py
```

- inference

```
python inference.py
```

## Arguments Usage

- RBERT

| Argument               | type  | Default                         | Explanation                                  |
| ---------------------- | ----- | ------------------------------- | -------------------------------------------- |
| batch_size             | int   | 40                              | 학습&예측에 사용될 batch size                |
| num_folds              | int   | 5                               | Stratified KFold의 fold 개수                 |
| num_train_epochs       | int   | 5                               | 학습 epoch                                   |
| loss                   | str   | focalloss                       | loss function                                |
| gamma                  | float | 1.0                             | focalloss 사용시 gamma 값                    |
| optimizer              | str   | adamp                           | 학습 optimizer                               |
| scheduler              | str   | get_cosine_schedule_with_warmup | learning rate를 조절하는 scheduler           |
| learning_rate          | float | 0.00005                         | 초기 learning rate 값                        |
| weight_decay           | float | 0.01                            | Loss function에 Weigth가 커질 경우 패널티 값 |
| warmup_step            | int   | 500                             |
| debug                  | bool  | false                           | 디버그 모드일 경우 True                      |
| dropout_rate           | float | 0.1                             | dropout 비율                                 |
| save_steps             | int   | 100                             | 모델 저장 step 수                            |
| evaluation_steps       | int   | 100                             | evaluation할 step 수                         |
| metric_for_best_model  | str   | eval/loss                       | 최고 성능을 가늠하는 metric                  |
| load_best_model_at_end | bool  | True                            |
