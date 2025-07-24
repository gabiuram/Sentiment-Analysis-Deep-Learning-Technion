import os
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
import zipfile
import torch
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup #hugging face imports
import torch.nn as nn
from sklearn import metrics

TOKENIZER_LSTM = Tokenizer(
    oov_token='<UNK>',
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    char_level=False
)
ATTRIBUTES = ['antagonize' , 'condescending', 'dismissive', 'generalisation',
    'hostile', 'sarcastic', 'unhealthy']
class UCC_Dataset_LSTM(torch.utils.data.Dataset):
    def __init__(self, data, max_length=512, vocab_size=10000, fit_tokenizer=True):
        self.data = data.copy()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer = TOKENIZER_LSTM
        self.attributes = ATTRIBUTES

        # Extract texts and labels
        self.texts = data['comment'].astype(str).tolist()
        self.labels = data[self.attributes].values.astype(np.float32)

        #fit_tokenizer should be set to True during training and False otherwise
        if fit_tokenizer:
            self._fit_tokenizer()

        # Tokenize all texts
        self.tokenized_texts = self._tokenize_texts()

    def _fit_tokenizer(self):
        self.tokenizer.num_words = self.vocab_size
        self.tokenizer.fit_on_texts(self.texts)

    def _tokenize_texts(self):
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(self.texts)

        # Pad sequences to max_length
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post',
            value=0  # padding value
        )

        return padded_sequences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get tokenized text
        input_ids = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != 0).long()

        # Get labels
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

TOKENIZER_PATH = "roberta-base"
TOKENIZER_BERT = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
MAX_TOKEN_LEN_BERT = 128

class UCC_Dataset_BERT(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data
    self.tokenizer = TOKENIZER_BERT
    self.attributes = ATTRIBUTES
    self.max_token_len = MAX_TOKEN_LEN_BERT

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    comment = str(self.data.iloc[idx].comment)
    # Explicitly cast to float to ensure numeric type
    attributes_labels = torch.tensor(self.data.loc[idx, self.attributes].values.astype(float), dtype=torch.float)
#https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus:~:text=tokenize(text)).-,encode_plus,-%3C
    tokenized_comment = self.tokenizer.encode_plus(
        comment,
        add_special_tokens = True,
        padding = 'max_length',
        truncation = True,
        max_length = self.max_token_len,
        return_tensors = 'pt',
        return_attention_mask = True)
    return {
        'input_ids': tokenized_comment['input_ids'].flatten(),
        'attention_mask': tokenized_comment['attention_mask'].flatten(),
        'attributes_labels': attributes_labels
    }

