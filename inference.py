# import dataset wrangler
import numpy as np
import pandas as pd

# import torch and its derivatives
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import third party library
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# import custom modules
from metrics import *
from dataset import *
from models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_CFG = {}
IB_CFG = {}
RBERT_CFG = {}
CONCAT_CFG = {}

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)

DATA_CFG = SAVED_CFG["data"]
IB_CFG = SAVED_CFG["IB"]
RBERT_CFG = SAVED_CFG["RBERT"]
CONCAT_CFG = SAVED_CFG["Concat"]


def num_to_label(label):
    origin_label = []
    with open("data/dict_label_to_num.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    new_dict = {value: key for key, value in dict_num_to_label.items()}
    for v in label:
        origin_label.append(new_dict[v])
    return origin_label


def inference_rbert():

    PORORO_TEST_PATH = DATA_CFG["pororo_test_path"]
    test_dataset = pd.read_csv(PORORO_TEST_PATH)
    # test_dataset = test_dataset.drop(test_dataset.columns[0], axis=1)
    test_dataset["label"] = 100
    print(len(test_dataset))
    MODEL_NAME = RBERT_CFG["pretrained_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    special_token_list = []
    with open(DATA_CFG["pororo_special_token_path"], "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(set(special_token_list))}
    )
    test_set = RBERT_Dataset(test_dataset, tokenizer, is_training=False)
    print(len(test_set))
    test_data_loader = DataLoader(
        test_set,
        batch_size=RBERT_CFG["batch_size"],
        num_workers=RBERT_CFG["num_workers"],
        shuffle=False,
    )
    oof_pred = []  # out of fold prediction list
    for i in range(5):
        model_path = "/opt/ml/klue-level2-nlp-15/notebooks/results/{}-fold-5-best-eval-loss-model.pt".format(
            i + 1
        )
        model = RBERT(
            RBERT_CFG["pretrained_model_name"], dropout_rate=RBERT_CFG["dropout_rate"]
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        output_pred = []
        for i, data in enumerate(tqdm(test_data_loader)):
            with torch.no_grad():
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    subject_mask=data["subject_mask"].to(device),
                    object_mask=data["object_mask"].to(device),
                    # token_type_ids=data['token_type_ids'].to(device) # RoBERTa does not use token_type_ids.
                )
            output_pred.extend(outputs.cpu().detach().numpy())
        output_pred = F.softmax(torch.Tensor(output_pred), dim=1)
        oof_pred.append(np.array(output_pred)[:, np.newaxis])

        # Prevent OOM error
        model.cpu()
        del model
        torch.cuda.empty_cache()

    models_prob = np.mean(
        np.concatenate(oof_pred, axis=2), axis=2
    )  # probability of each class
    result = np.argmax(models_prob, axis=-1)  # label
    # print(result, type(result))
    # print(models_prob.shape, list_prob)

    list_prob = models_prob.tolist()

    test_id = test_dataset["id"]
    df_submission = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": num_to_label(result),
            "probs": list_prob,
        }
    )
    df_submission.to_csv("./prediction/submission_RBERT.csv", index=False)
