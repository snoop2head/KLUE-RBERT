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

## References

- [monologg's R-BERT Implementation in Pytorch](https://github.com/monologg/R-BERT)
- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284?context=cs)

## Authorship

- [jjonhwa](https://github.com/jjonhwa)
- [🤚 snoop2head](https://github.com/snoop2head)
- [kimyeondu](kimyeondu)
- [hihellohowareyou](https://github.com/hihellohowareyou)
- [shawnhyeonsoo](https://github.com/shawnhyeonsoo)
- [danielkim30433](https://github.com/danielkim30433)
- [ntommy11](https://github.com/ntommy11)
