import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import AutoTokenizer

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
    """PyTorch Dataset for LSTM-based sentiment classification."""

    def __init__(self, data, max_length=512, vocab_size=10000, fit_tokenizer=True):
        """
        Initialize the LSTM dataset.

        Args:
            data (pd.DataFrame): Input dataframe containing 'comment' and label columns.
            max_length (int, optional): Maximum sequence length for padding/truncating. Defaults to 512.
            vocab_size (int, optional): Maximum vocabulary size for tokenizer. Defaults to 10000.
            fit_tokenizer (bool, optional): Whether to fit tokenizer on provided data. Defaults to True.
        """
        self.data = data.copy()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer = TOKENIZER_LSTM
        self.attributes = ATTRIBUTES

        self.texts = data['comment'].astype(str).tolist()
        self.labels = data[self.attributes].values.astype(np.float32)

        if fit_tokenizer:
            self._fit_tokenizer()

        self.tokenized_texts = self._tokenize_texts()

    def _fit_tokenizer(self):
        """Fit the tokenizer on the dataset texts."""
        self.tokenizer.num_words = self.vocab_size
        self.tokenizer.fit_on_texts(self.texts)

    def _tokenize_texts(self):
        """
        Tokenize and pad the dataset texts.

        Returns:
            np.ndarray: Tokenized and padded sequences.
        """
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post',
            value=0
        )
        return padded_sequences

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        input_ids = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
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
    """PyTorch Dataset for BERT-based sentiment classification."""

    def __init__(self, data):
        """
        Initialize the BERT dataset.

        Args:
            data (pd.DataFrame): Input dataframe containing 'comment' and label columns.
        """
        self.data = data
        self.tokenizer = TOKENIZER_BERT
        self.attributes = ATTRIBUTES
        self.max_token_len = MAX_TOKEN_LEN_BERT

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing tokenized input_ids, attention_mask, and labels.
        """
        comment = str(self.data.iloc[idx].comment)
        labels = torch.tensor(self.data.loc[idx, self.attributes].values.astype(float), dtype=torch.float)
        tokenized_comment = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_tensors='pt',
            return_attention_mask=True
        )
        return {
            'input_ids': tokenized_comment['input_ids'].flatten(),
            'attention_mask': tokenized_comment['attention_mask'].flatten(),
            'labels': labels
        }
