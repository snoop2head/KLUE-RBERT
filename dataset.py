# import python innate functions
import random
import pickle
from ast import literal_eval
from collections import defaultdict

# import dataset wrangler
import numpy as np
import pandas as pd

# import machine learning modules
from sklearn.model_selection import StratifiedKFold

# import torch and its applications
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# import from huggingface transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM


# import third party modules
import yaml

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


class RBERT_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, is_training: bool = True):

        # pandas.Dataframe dataset
        self.dataset = dataset
        self.sentence = self.dataset["sentence"]
        self.subject_entity = self.dataset["subject_entity"]
        self.object_entity = self.dataset["object_entity"]
        if is_training:
            self.train_label = label_to_num(self.dataset["label"].values)
        if not is_training:
            self.train_label = self.dataset["label"].values
        self.label = torch.tensor(self.train_label)

        # tokenizer and etc
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        subject_entity = self.subject_entity[idx]
        object_entity = self.object_entity[idx]
        label = self.label[idx]

        # concat entity in the beginning
        concat_entity = subject_entity + "[SEP]" + object_entity

        # tokenize
        item = self.tokenizer(
            concat_entity,
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=RBERT_CFG.max_token_length,
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )

        # RoBERTa's provided masks (do not include token_type_ids for RoBERTa)
        item["input_ids"] = item["input_ids"].squeeze(0)
        item["attention_mask"] = item["attention_mask"].squeeze(0)

        # add subject and object entity masks where masks notate where the entity is
        subject_entity_mask, object_entity_mask = self.add_entity_mask(
            item, subject_entity, object_entity
        )
        item["subject_mask"] = torch.Tensor(subject_entity_mask)
        item["object_mask"] = torch.Tensor(object_entity_mask)

        # fill label
        item["label"] = label
        return item

    def __len__(self):
        return len(self.dataset)

    def add_entity_mask(self, item, subject_entity, object_entity):
        """add entity token to input_ids"""
        # print("tokenized input ids: \n",item['input_ids'])

        # initialize entity masks
        subject_entity_mask = np.zeros(RBERT_CFG.max_token_length, dtype=int)
        object_entity_mask = np.zeros(RBERT_CFG.max_token_length, dtype=int)

        # get token_id from encoding subject_entity and object_entity
        subject_entity_token_ids = self.tokenizer.encode(
            subject_entity, add_special_tokens=False
        )
        object_entity_token_ids = self.tokenizer.encode(
            object_entity, add_special_tokens=False
        )
        # print("entity token's input ids: ",subject_entity_token_ids, object_entity_token_ids)

        # get the length of subject_entity and object_entity
        subject_entity_length = len(subject_entity_token_ids)
        object_entity_length = len(object_entity_token_ids)

        # find coordinates of subject_entity_token_ids inside the item["input_ids"]
        subject_coordinates = np.where(item["input_ids"] == subject_entity_token_ids[0])
        subject_coordinates = list(
            map(int, subject_coordinates[0])
        )  # change the subject_coordinates into int type
        for subject_index in subject_coordinates:
            subject_entity_mask[
                subject_index : subject_index + subject_entity_length
            ] = 1

        # find coordinates of object_entity_token_ids inside the item["input_ids"]
        object_coordinates = np.where(item["input_ids"] == object_entity_token_ids[0])
        object_coordinates = list(
            map(int, object_coordinates[0])
        )  # change the object_coordinates into int type
        for object_index in object_coordinates:
            object_entity_mask[object_index : object_index + object_entity_length] = 1

        # print(subject_entity_mask)
        # print(object_entity_mask)

        return subject_entity_mask, object_entity_mask
