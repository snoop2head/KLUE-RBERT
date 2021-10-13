import pandas as pd
import numpy as np
import pickle as pickle


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


FIRST = "/opt/ml/code/output(1).csv"
SECOND = "/opt/ml/code/output(2).csv"
THRID = "/opt/ml/code/output(3).csv"

# csv to dataframe
f = pd.read_csv(FIRST)
s = pd.read_csv(SECOND)
t = pd.read_csv(THRID)


f_prob = np.array(list(map(eval, f["probs"])))
s_prob = np.array(list(map(eval, s["probs"])))
t_prob = np.array(list(map(eval, t["probs"])))

# how to soft voting
probs_triple = (f_prob + s_prob + t_prob) / 3
probs_double = (f_prob + s_prob) / 2
probs_weighted_double = (f_prob * 1.2 + s_prob * 0.8) / 2

# extract prediction
pred_triple = np.argmax(probs_triple, axis=-1)
pred_double = np.argmax(probs_double, axis=-1)
pred_weighted_double = np.argmax(probs_weighted_double, axis=-1)

# index
id = list(range(len(pred)))

# num to label
pred_triple_answer = num_to_label(pred_triple)
pred_double_answer = num_to_label(pred_double)
pred_weighted_double_answer = num_to_label(pred_weighted_double)

# numpy to list -> Because it needs to be input into Dataframe
probs_triple = probs_triple.tolist()
probs_double = probs_double.tolist()
probs_weighted_double = probs_weighted_double.tolist()

# dataframe to scv
output = pd.DataFrame(
    {
        "id": id,
        "pred_label": pred_triple_answer,
        "probs": probs_triple,
    }
)
output.to_csv("./prediction/submission_triple.csv", index=False)

output_s = pd.DataFrame(
    {
        "id": id,
        "pred_label": pred_double_answer,
        "probs": probs_double,
    }
)
output_s.to_csv("./prediction/submission_double.csv", index=False)

output_w = pd.DataFrame(
    {
        "id": id,
        "pred_label": pred_weighted_double_answer,
        "probs": probs_weighted_double,
    }
)
output_w.to_csv("./prediction/submission_weighted_double.csv", index=False)
