import torch
from torch import nn
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


class FCLayer(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""

    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""

    def __init__(
        self,
        model_name: str = "klue/roberta-large",
        num_labels: int = 30,
        dropout_rate: float = 0.1,
        special_tokens_dict: dict = None,
        is_train: bool = True,
    ):
        super(RBERT, self).__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.backbone_model = AutoModel.from_pretrained(model_name, config=config)
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        config.num_labels = num_labels

        # add special tokens
        self.special_tokens_dict = special_tokens_dict
        self.backbone_model.resize_token_embeddings(len(self.tokenizer))

        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids,
        attention_mask,
        subject_mask=None,
        object_mask=None,
        labels=None,
    ):

        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs[
            "pooler_output"
        ]  # [CLS] token's hidden featrues(hidden state)

        # hidden state's average in between entities
        # print(sequence_output.shape, subject_mask.shape)
        e1_h = self.entity_average(
            sequence_output, subject_mask
        )  # token in between subject entities ->
        e2_h = self.entity_average(
            sequence_output, object_mask
        )  # token in between object entities

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(
            pooled_output
        )  # [CLS] token -> hidden state | green on diagram
        e1_h = self.entity_fc_layer(
            e1_h
        )  # subject entity's fully connected layer | yellow on diagram
        e2_h = self.entity_fc_layer(
            e2_h
        )  # object entity's fully connected layer | red on diagram

        # Concat -> fc_layer / [CLS], subject_average, object_average
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        return logits

        # WILL USE FOCAL LOSS INSTEAD OF MSELoss and CrossEntropyLoss
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #     outputs = (loss,) + outputs

        #  return outputs  # (loss), logits, (hidden_states), (attentions)
