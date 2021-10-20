# RBERT for Relation Extraction task for KLUE

## Hardware

- `GPU : Tesla V100 32GB`

## Project Description

Relation Extraction task is one of [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE)'s task. 

Korean Language Understanding Evaluation(KLUE) Benchmark is composed of 8 tasks:

- Topic Classification (TC)
- Sentence Textual Similarity (STS)
- Natural Language Inference (NLI)
- Named Entity Recognition (NER)
- **Relation Extraction (RE)**
- (Part-Of-Speech) + Dependency Parsing (DP)
- Machine Reading Comprehension (MRC)
- Dialogue State Tracking (DST)

This repo contains custom dataset, custom training code utilizing [monologg's R-BERT Implementation](https://github.com/monologg/R-BERT).


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
