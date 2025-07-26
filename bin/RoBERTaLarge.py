import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn import metrics
from peft import get_peft_model, LoraConfig, TaskType
from src.sentiment_analysis.datasets import UCC_Dataset_BERT
from src.sentiment_analysis.utils import Training, ATTRIBUTES, ATTRIBUTES_MERGED

TOKENIZER_PATH = "roberta-large"
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
MODEL_PATH = 'roberta-large'
HEALTHY_SAMPLE = 5000
MAX_TOKEN_LEN = 64
BATCH_SIZE = 64
NUM_EPOCHS = 21
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attributes = [
    'antagonize' , 'condescending', 'dismissive', 'generalisation',
    'hostile', 'sarcastic', 'healthy']

class UCC_classifier(nn.Module):
  def __init__(self):
    super(UCC_classifier, self).__init__()
    config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=len(ATTRIBUTES), problem_type="multi_label_classification")
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    self.roberta = get_peft_model(base_model, lora_config)
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(config.hidden_size // 2, len(ATTRIBUTES))
    )
    nn.init.xavier_uniform_(self.fc[0].weight)
    nn.init.xavier_uniform_(self.fc[-1].weight)

    for name, param in self.roberta.named_parameters():
      if 'lora' not in name:
        param.requires_grad = False

    for param in self.fc.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    x = self.roberta.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    x = torch.mean(x, 1)
    x = self.dropout(x)
    x = self.fc(x)
    return x

def train_model(model, train_loader, val_loader, num_epochs = NUM_EPOCHS):
  best_auc = 0
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  train_loss_per_epoch = []
  val_loss_per_epoch = []

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100
  )

  best_val_loss = float('inf')
  patience = 10
  patience_counter = 0
  completed_epochs = 0
  clipping_value = 0.1

  for epoch in range(1, num_epochs + 1):
      model.train()
      running_loss = 0.0
      epoch_time = time.time()
      for batch_data in train_loader:
          inputs = batch_data['input_ids'].to(device)
          attention_mask = batch_data['attention_mask'].to(device)
          labels = batch_data['labels'].to(device)

          outputs = model(inputs, attention_mask)
          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
          optimizer.step()
          scheduler.step()

          running_loss += loss.data.item()

      running_loss /= len(train_loader)
      train_loss_per_epoch.append(running_loss)

      val_loss = Training.evaluate_model(model, val_loader, criterion, device)
      val_loss_per_epoch.append(val_loss)
      completed_epochs = epoch

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
      labels = np.array(test_data[ATTRIBUTES])

      auc_scores = []
      for i in range(len(ATTRIBUTES)):
          try:
              auc = metrics.roc_auc_score(labels[:, i].astype(int), predictions[:, i])
              auc_scores.append(auc)
          except ValueError:
              print(f"Could not calculate AUC for attribute {ATTRIBUTES[i]}")
              pass

      average_auc = np.mean(auc_scores) if auc_scores else 0

      if average_auc > best_auc:
          best_auc = average_auc
          patience_counter = 0
          torch.save(model.state_dict(), 'best_model.pth')
      else:
          patience_counter += 1

      if patience_counter >= patience:
          print("Early stopping triggered. Training stopped.")
          break

      log = "Epoch: {} | Train Loss: {:.4f} | Val Loss: {:.4f} | ".format(epoch, running_loss, val_loss)
      epoch_time = time.time() - epoch_time
      log += "Epoch Time: {:.2f} secs".format(epoch_time)
      print(log)

      torch.cuda.empty_cache()

  print('==> Finished Training ...')

  plt.figure(figsize=(8, 5))
  plt.plot(range(1, completed_epochs + 1), train_loss_per_epoch, label='Train Loss')
  plt.plot(range(1, completed_epochs + 1), val_loss_per_epoch, label='Val Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Curve')
  plt.legend()
  plt.tight_layout()
  plt.savefig("Final.png")
  plt.show()

if __name__ == '__main__':
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    val_data = pd.read_csv('data/val.csv')

    train_data[attributes].sum().plot(kind='bar')
    plt.title('Training Samples per Attribute')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    healthy_samples = train_data[train_data['healthy'] == 1]
    unhealthy_samples = train_data[train_data['healthy'] == 0]

    print(f"Number of healthy training samples before preprocessing: {len(healthy_samples)}")
    print(f"Number of unhealthy training samples before preprocessing: {len(unhealthy_samples)}")

    balanced_train_data = Training.preprocess_train(train_data, HEALTHY_SAMPLE, attributes, ATTRIBUTES_MERGED)
    healthy_samples = balanced_train_data[balanced_train_data['healthy'] == 1]
    unhealthy_samples = balanced_train_data[balanced_train_data['unhealthy'] == 1]

    print(f"Number of healthy training samples after preprocessing: {len(healthy_samples)}")
    print(f"Number of unhealthy training samples after preprocessing: {len(unhealthy_samples)}")

    model = UCC_classifier()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    train_ds, val_ds, train_loader, val_loader, test_loader = Training.load_data(
        balanced_train_data, val_data, test_data, BATCH_SIZE, UCC_Dataset_BERT
    )

    train_model(model, train_loader, val_loader)

    labels = np.array(test_data[ATTRIBUTES])
    model_path = 'best_model.pth'
    Training.evaluate_saved_model(
        model_path, test_loader, test_data,
        "RoBERTa-large + LoRA\nROC-AUC Score on Test Set",
        True, UCC_classifier
    )
