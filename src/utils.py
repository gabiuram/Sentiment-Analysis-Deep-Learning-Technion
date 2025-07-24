"""Functions used for multiple models."""
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.datasets import UCC_Dataset_BERT, UCC_Dataset_LSTM


class Training:

    #balance training data by removing healthy labels without hurting other attributes
    @staticmethod
    def preprocess_train(train_data, healthy_sample, attributes, attributes_merged):
      healthy_symptomatic = train_data[
          (train_data['healthy'] == 1) &
          (train_data[attributes].sum(axis=1) > 1)
      ]

      healthy_clean = train_data[
          (train_data['healthy'] == 1) &
          (train_data[attributes].sum(axis=1) == 1)
      ]

      unhealthy = train_data[train_data['healthy'] == 0]

      sample_size = len(unhealthy) - len(healthy_symptomatic) + healthy_sample #need more healthy samples to reflect distribution

      healthy_clean = healthy_clean.sample(n=sample_size, random_state=42)

      balanced_train_data = pd.concat([healthy_symptomatic, healthy_clean, unhealthy])

      #shuffle the dataset
      balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

      #add additional unhealthy label for comparative purposes: https://github.com/conversationai/unhealthy-conversations?tab=readme-ov-file
      balanced_train_data['unhealthy'] = 1 - balanced_train_data['healthy']

      balanced_train_data[attributes_merged].sum().plot(kind='bar')
      plt.title('Training Samples per Attribute After Preprocessing')
      plt.ylabel('Count')
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()

      return balanced_train_data

    @staticmethod
    def load_data(train_data, val_data, test_data, batch_size, dataset_class):
        # Add the 'unhealthy' column to val_data
        val_data['unhealthy'] = 1 - val_data['healthy']
        test_data['unhealthy'] = 1 - test_data['healthy']

        if dataset_class.__name__ == "UCC_Dataset_BERT":
            train_ds = UCC_Dataset_BERT(train_data)
            val_ds = UCC_Dataset_BERT(val_data)
            test_ds = UCC_Dataset_BERT(test_data)
        else:
            train_ds = UCC_Dataset_LSTM(train_data)
            val_ds = UCC_Dataset_LSTM(val_data, fit_tokenizer=False)
            test_ds = UCC_Dataset_LSTM(test_data, fit_tokenizer=False)


        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False
        )

        return train_ds, val_ds, train_loader, val_loader, test_loader

    @staticmethod
    def print_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print(f"model size: {size_all_mb:.2f} MB")

