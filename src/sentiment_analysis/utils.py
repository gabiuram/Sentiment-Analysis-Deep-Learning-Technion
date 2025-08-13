"""Functions used for multiple models."""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sentiment_analysis.datasets import UCC_Dataset_BERT, UCC_Dataset_LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ATTRIBUTES = ['antagonize' , 'condescending', 'dismissive', 'generalisation',
    'hostile', 'sarcastic', 'unhealthy']
ATTRIBUTES_MERGED = ['antagonize' , 'condescending', 'dismissive', 'generalisation',
    'hostile', 'sarcastic', 'unhealthy', 'healthy']


class Training:
    """Utility class containing preprocessing, data loading, and evaluation functions for sentiment analysis models."""

    @staticmethod
    def preprocess_train(train_data, healthy_sample, attributes, attributes_merged):
        """
        Balance the training dataset by sampling healthy labels without reducing other attribute diversity.

        Args:
            train_data (pd.DataFrame): Original training dataframe.
            healthy_sample (int): Additional number of healthy samples to include.
            attributes (list): List of target attribute column names.
            attributes_merged (list): List of attributes including 'healthy'.

        Returns:
            pd.DataFrame: Balanced and shuffled training dataframe.
        """
        healthy_symptomatic = train_data[
            (train_data['healthy'] == 1) &
            (train_data[attributes].sum(axis=1) > 1)
        ]

        healthy_clean = train_data[
            (train_data['healthy'] == 1) &
            (train_data[attributes].sum(axis=1) == 1)
        ]

        unhealthy = train_data[train_data['healthy'] == 0]

        sample_size = len(unhealthy) - len(healthy_symptomatic) + healthy_sample
        healthy_clean = healthy_clean.sample(n=sample_size, random_state=42)

        balanced_train_data = pd.concat([healthy_symptomatic, healthy_clean, unhealthy])
        balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

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
        """
        Create dataset objects and data loaders for training, validation, and testing.

        Args:
            train_data (pd.DataFrame): Training dataframe.
            val_data (pd.DataFrame): Validation dataframe.
            test_data (pd.DataFrame): Test dataframe.
            batch_size (int): Batch size for data loaders.
            dataset_class (type): Dataset class to use (UCC_Dataset_BERT or UCC_Dataset_LSTM).

        Returns:
            tuple: (train_ds, val_ds, train_loader, val_loader, test_loader)
        """
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
        """
        Print the size of a PyTorch model in MB.

        Args:
            model (torch.nn.Module): The model to evaluate.
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print(f"model size: {size_all_mb:.2f} MB")

    @staticmethod
    def evaluate_saved_model(model_path, test_loader, test_data, plot_name, LLM=False, classifier=None):
        """
        Load a saved model and evaluate it on the test set, plotting ROC curves.

        Args:
            model_path (str): Path to the saved model file.
            test_loader (DataLoader): DataLoader for the test set.
            test_data (pd.DataFrame): Test dataframe containing labels.
            plot_name (str): Title for the ROC plot.
            LLM (bool, optional): Whether the model is a language model requiring a classifier wrapper. Defaults to False.
            classifier (type, optional): Classifier class to initialize if LLM is True.
        """
        labels = np.array(test_data[ATTRIBUTES])
        if LLM:
            model = classifier()
            model.load_state_dict(torch.load(model_path, weights_only=False, map_location=torch.device("cpu")))
        else:
            model = torch.load(model_path, weights_only=False, map_location=torch.device("cpu"))

        model.eval()
        predictions = []
        model.to(device)

        with torch.no_grad():
            for batch_data in test_loader:
                comments = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                outputs = model(comments, attention_mask)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions)

        print("Printing Results")
        plt.figure(figsize=(15, 8))
        for i, attribute in enumerate(ATTRIBUTES):
            fpr, tpr, _ = metrics.roc_curve(
                labels[:, i].astype(int), predictions[:, i])
            auc = metrics.roc_auc_score(
                labels[:, i].astype(int), predictions[:, i])
            plt.plot(fpr, tpr, label='%s %g' % (attribute, auc))
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.legend(loc='lower right')
        plt.title(plot_name, fontsize=14, fontweight='bold')
        plt.savefig("Results.png")
        plt.show()

    @staticmethod
    def evaluate_model(model, val_loader, criterion, device):
        """
        Evaluate a model on the validation set and return the loss.

        Args:
            model (torch.nn.Module): Model to evaluate.
            val_loader (DataLoader): DataLoader for the validation set.
            criterion: Loss function to compute validation loss.
            device (torch.device): Device to perform computation on.

        Returns:
            float: Average validation loss.
        """
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                comments = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                attributes = batch_data['labels'].to(device)

                outputs = model(comments, attention_mask)
                loss = criterion(outputs, attributes)
                val_loss += loss.item() * comments.size(0)

        val_loss /= len(val_loader.dataset)
        return val_loss
