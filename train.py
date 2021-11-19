# import python innate modules
import random
import os

# import data wrangling modules
import pandas as pd
import numpy as np

# import machine learning modules
from sklearn.metrics import f1_score, confusion_matrix

# import torch and its applications
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

# import transformers and its applications
from transformers import (
    AutoTokenizer,
    AdamW,
    get_cosine_schedule_with_warmup,
)

# import third party modules
import yaml
from tqdm import tqdm
from easydict import EasyDict
from adamp import AdamP

# import custom modules
from dataset import *
from models import *
from metrics import *
from loss import *


class dotdict(dict):
    """dot.notation access to dictionary attributes, as dict.key_name, not as dict["key_name"] """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    dotdict(SAVED_CFG)

DATA_CFG = dotdict(SAVED_CFG["data"])
RBERT_CFG = dotdict(SAVED_CFG["RBERT"])


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = True


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def train_rbert():
    # read pororo dataset
    df_pororo_dataset = pd.read_csv(DATA_CFG.pororo_train_path)

    # remove the first index column
    df_pororo_dataset = df_pororo_dataset.drop(df_pororo_dataset.columns[0], axis=1)

    # fetch tokenizer
    tokenizer = AutoTokenizer.from_pretrained(RBERT_CFG.pretrained_model_name)

    # fetch special tokens annotated with ner task
    special_token_list = []
    with open(DATA_CFG.pororo_special_token_path, "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(set(special_token_list))}
    )
    # print special tokens
    print(tokenizer.special_tokens_map)

    if torch.cuda.is_available() and RBERT_CFG.debug == False:
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("GPU Name:", torch.cuda.get_device_name(0))
        print(os.system("nvidia-smi"))
    else:
        print("No GPU, using CPU.")
        device = torch.device("cpu")

    train_data = RBERT_Dataset(df_pororo_dataset, tokenizer)
    dev_data = RBERT_Dataset(df_pororo_dataset, tokenizer)
    # print(df_pororo_dataset.head())

    stf = StratifiedKFold(
        n_splits=RBERT_CFG.num_folds, shuffle=True, random_state=seed_everything(42)
    )

    # criterion = FocalLoss(gamma=RBERT_CFG.gamma)  # 0.0 equals to CrossEntropy
    criterion = nn.CrossEntropyLoss()

    early_stop = 0

    for fold_num, (train_idx, dev_idx) in enumerate(
        stf.split(df_pororo_dataset, list(df_pororo_dataset["label"]))
    ):

        print(f"#################### Fold: {fold_num + 1} ######################")

        train_set = Subset(train_data, train_idx)
        dev_set = Subset(dev_data, dev_idx)
        # print(train_set[0])
        # print(train_set[0]["subject_mask"])
        # print(train_set[0]["object_mask"])
        # print(train_set[0]["label"])

        train_loader = DataLoader(
            train_set,
            batch_size=RBERT_CFG.batch_size,
            shuffle=True,
            num_workers=RBERT_CFG.num_workers,
        )
        dev_loader = DataLoader(
            dev_set,
            batch_size=RBERT_CFG.batch_size,
            shuffle=False,
            num_workers=RBERT_CFG.num_workers,
        )

        # fetch model
        model = RBERT(RBERT_CFG.pretrained_model_name)
        model.to(device)

        # fetch loss function, optimizer, scheduler outside of torch library
        # https://github.com/clovaai/AdamP
        optimizer = AdamP(
            model.parameters(),
            lr=RBERT_CFG.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=RBERT_CFG.weight_decay,
        )
        # optimizer = AdamW(model.parameters(), lr=RBERT_CFG.learning_rate, betas=(0.9, 0.999), weight_decay=CFG.weight_decay) # AdamP is better

        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=RBERT_CFG.warmup_steps,
            num_training_steps=len(train_loader) * RBERT_CFG.num_train_epochs,
        )

        best_eval_loss = 1.0
        steps = 0

        # fetch training loop
        for epoch in tqdm(range(RBERT_CFG.num_train_epochs)):
            train_loss = Metrics()
            dev_loss = Metrics()
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                # print(item)
                # assign forward() arguments to the device
                # print(batch)
                # print(batch["input_ids"].shape)

                label = batch["label"].to(device)
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "subject_mask": batch["subject_mask"].to(device),
                    # 'token_type_ids' # NOT FOR ROBERTA!
                    "object_mask": batch["object_mask"].to(device),
                    "label": label,
                }

                # model to training mode
                model.train()
                pred_logits = model(**inputs)
                loss = criterion(pred_logits, label)
                # print(loss)

                # backward
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update metrics
                train_loss.update(loss.item(), len(label))
                # print(train_loss)

                steps += 1
                # for every 100 steps
                if steps % 10 == 0:
                    print(
                        "Epoch: {}/{}".format(epoch + 1, RBERT_CFG.num_train_epochs),
                        "Step: {}".format(steps),
                        "Train Loss: {:.4f}".format(train_loss.avg),
                    )
                    for dev_batch in dev_loader:
                        dev_label = dev_batch["label"].to(device)
                        dev_inputs = {
                            "input_ids": dev_batch["input_ids"].to(device),
                            "attention_mask": dev_batch["attention_mask"].to(device),
                            "subject_mask": dev_batch["subject_mask"].to(device),
                            # 'token_type_ids' # NOT FOR ROBERTA!
                            "object_mask": dev_batch["object_mask"].to(device),
                            "label": dev_label,
                        }

                        # switch model to eval mode
                        model.eval()
                        dev_pred_logits = model(**dev_inputs)
                        loss = criterion(dev_pred_logits, dev_label)

                        # update metrics
                        dev_loss.update(loss.item(), len(dev_label))

                    # print metrics
                    print(
                        "Epoch: {}/{}".format(epoch + 1, RBERT_CFG.num_train_epochs),
                        "Step: {}".format(steps),
                        "Dev Loss: {:.4f}".format(dev_loss.avg),
                    )

                    if best_eval_loss > dev_loss.avg:
                        best_eval_loss = dev_loss.avg
                        torch.save(
                            model.state_dict(),
                            "./results/{}-fold-{}-best-eval-loss-model.pt".format(
                                fold_num + 1, RBERT_CFG.num_folds
                            ),
                        )
                        print(
                            "Saved model with lowest validation loss: {:.4f}".format(
                                best_eval_loss
                            )
                        )
                        early_stop = 0
                    else:
                        early_stop += 1
                        if early_stop > 2:
                            break

        # Prevent OOM error
        model.cpu()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_rbert()
