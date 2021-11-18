# import python innate modules
import os
import pickle as pickle

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
from easydict import EasyDict
import yaml


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


def label_to_num(label=None):
    num_label = []
    with open(DATA_CFG.label_to_num_file_path, "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


class RBERT_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, is_training: bool = True):

        print(dataset.head())

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

        # set tokenizer
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        print(sentence)
        subject_entity = self.subject_entity[idx]
        object_entity = self.object_entity[idx]
        label = self.label[idx]

        # concat entity in the beginning
        concat_entity = subject_entity + "[SEP]" + object_entity

        # tokenize
        encoded_dict = self.tokenizer(
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
        encoded_dict["input_ids"] = encoded_dict["input_ids"].squeeze(0)
        encoded_dict["attention_mask"] = encoded_dict["attention_mask"].squeeze(0)

        print(
            "encoded-dict",
            self.tokenizer.convert_ids_to_tokens(encoded_dict["input_ids"]),
        )

        # add subject and object entity masks where masks notate where the entity is
        subject_entity_mask, object_entity_mask = self.add_entity_mask(
            encoded_dict, subject_entity, object_entity
        )
        encoded_dict["subject_mask"] = subject_entity_mask
        encoded_dict["object_mask"] = object_entity_mask

        # fill label
        encoded_dict["label"] = label
        return encoded_dict

    def __len__(self):
        return len(self.dataset)

    def add_entity_mask(self, encoded_dict, subject_entity, object_entity):
        """add entity token to input_ids"""
        print("tokenized input ids: \n", encoded_dict["input_ids"])

        # initialize entity masks
        print(RBERT_CFG.max_token_length)
        subject_entity_mask = np.zeros(RBERT_CFG.max_token_length, dtype=int)
        object_entity_mask = np.zeros(RBERT_CFG.max_token_length, dtype=int)
        print("subject_entity_mask: \n", subject_entity_mask)
        print("object_entity_mask: \n", object_entity_mask)

        # get token_id from encoding subject_entity and object_entity
        print("subject_entity: \n", subject_entity)
        print("object_entity: \n", object_entity)
        subject_entity_token_ids = self.tokenizer.encode(
            subject_entity, add_special_tokens=False
        )
        object_entity_token_ids = self.tokenizer.encode(
            object_entity, add_special_tokens=False
        )
        print(
            "entity token's input ids: ",
            subject_entity_token_ids,
            object_entity_token_ids,
        )

        # get the length of subject_entity and object_entity
        subject_entity_length = len(subject_entity_token_ids)
        object_entity_length = len(object_entity_token_ids)

        # find coordinates of subject_entity_token_ids inside the encoded_dict["input_ids"]
        subject_coordinates = np.where(
            encoded_dict["input_ids"] == subject_entity_token_ids[1]
        )
        # change the subject_coordinates into int type
        subject_coordinates = list(map(int, subject_coordinates[0]))

        print("subject_coordinates: ", subject_coordinates)
        for subject_index in subject_coordinates:
            subject_entity_mask[
                subject_index : subject_index + subject_entity_length
            ] = 1
        # find coordinates of object_entity_token_ids inside the encoded_dict["input_ids"]
        object_coordinates = np.where(
            encoded_dict["input_ids"] == object_entity_token_ids[1]
        )
        object_coordinates = list(
            map(int, object_coordinates[0])
        )  # change the object_coordinates into int type

        print("object_coordinates", object_coordinates)
        for object_index in object_coordinates:
            object_entity_mask[object_index : object_index + object_entity_length] = 1
        print(subject_entity_mask)
        print(object_entity_mask)

        return torch.Tensor(subject_entity_mask), torch.Tensor(object_entity_mask)

