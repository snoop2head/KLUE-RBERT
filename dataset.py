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

        self.dataset = dataset  # pandas.Dataframe dataset
        self.sentence = self.dataset["sentence"]
        self.subject_entity = self.dataset["subject_entity"]
        self.object_entity = self.dataset["object_entity"]
        if is_training:
            self.train_label = label_to_num(self.dataset["label"].values)
        if not is_training:
            self.train_label = self.dataset["label"].values
        self.label = torch.tensor(self.train_label)

        self.tokenizer = tokenizer  # set tokenizer
        self.list_additional_special_tokens = tokenizer.special_tokens_map[
            "additional_special_tokens"
        ]

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
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

        # notate where the subject and object entity as separate entity attention mask
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
        """
        based on special token's coordinate, 
        make attention mask for subject and object entities' location 

        Variables:
        - sentence: 그는 [SUB-ORGANIZATION]아메리칸 리그[/SUB-ORGANIZATION]가 출범한 [OBJ-DATE]1901년[/OBJ-DATE] 당시 .426의 타율을 기록하였다.
        - encoded_dict: ['[CLS]', "'", '[SUB-ORGANIZATION]', '아메리칸', '리그', '[/SUB-ORGANIZATION]', "'", '[SEP]', "'", '[OBJ-DATE]', '190', '##1', '##년', '[/OBJ-DATE]', "'", '[SEP]', '그', '##는', '[SUB-ORGANIZATION]', '아메리칸', '리그', '[/SUB-ORGANIZATION]', '가', '출범', '##한', '[OBJ-DATE]', '190', '##1', '##년', '[/OBJ-DATE]', '당시', '.', '42', '##6', '##의', '타율', '##을', '기록', '##하', '##였', '##다', '.', '[SEP]', ]
        - subject_entity: ['[SUB-ORGANIZATION]', '아메리칸', '리그', '[/SUB-ORGANIZATION]']
        - subject_coordinates: index of the first [SUB-{}] added_special_tokens = [2, 18]
        - subject_entity_mask: [0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...]
        - object_entity: ['[OBJ-DATE]', '190', '##1', '##년', '[/OBJ-DATE]']
        - object_coordinates: index of the first [OBJ-{}] added_special_tokens = [9, 25]
        - object_entity_mask: [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...]

        Based on special tokens([SUB-ORGANIZATION], [OBJ-DATE]) for each entities, 1 in attention mask annotates the location of the entity.
        For more description, please refer to https://snoop2head.github.io/Relation-Extraction-Code/
        """

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

        # get the length of subject_entity and object_entity
        subject_entity_length = len(subject_entity_token_ids)
        object_entity_length = len(object_entity_token_ids)

        # find coordinates of subject_entity_token_ids based on special tokens
        subject_coordinates = np.where(
            encoded_dict["input_ids"] == subject_entity_token_ids[1]
        )

        # change the subject_coordinates into int type
        subject_coordinates = list(map(int, subject_coordinates[0]))

        # notate the location as 1 in subject_entity_mask
        for subject_index in subject_coordinates:
            subject_entity_mask[
                subject_index : subject_index + subject_entity_length
            ] = 1

        # find coordinates of subject_entity_token_ids based on special tokens
        object_coordinates = np.where(
            encoded_dict["input_ids"] == object_entity_token_ids[1]
        )

        # change the object_coordinates into int type
        object_coordinates = list(map(int, object_coordinates[0]))

        # notate the location as 1 in object_entity_mask
        for object_index in object_coordinates:
            object_entity_mask[object_index : object_index + object_entity_length] = 1

        return torch.Tensor(subject_entity_mask), torch.Tensor(object_entity_mask)

