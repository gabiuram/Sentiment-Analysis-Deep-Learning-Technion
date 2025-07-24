import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel #hugging face imports
from sklearn import metrics
from sentiment_analysis.datasets import UCC_Dataset_BERT
from sentiment_analysis.utils import Training, ATTRIBUTES, ATTRIBUTES_MERGED

TOKENIZER_PATH = "roberta-base"
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
MODEL_PATH = 'distilroberta-base'
HEALTHY_SAMPLE = 5000
MAX_TOKEN_LEN = 128
BATCH_SIZE = 256
NUM_EPOCHS = 21
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attributes = [
    'antagonize' , 'condescending', 'dismissive', 'generalisation',
    'hostile', 'sarcastic', 'healthy']

class UCC_classifier(nn.Module):
  def __init__(self):
    super(UCC_classifier, self).__init__()
    self.roBERTa = AutoModel.from_pretrained(MODEL_PATH, return_dict=True)
    self.dropout = nn.Dropout(0.4)
    self.fc = nn.Sequential(
        nn.Linear(self.roBERTa.config.hidden_size, self.roBERTa.config.hidden_size),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(self.roBERTa.config.hidden_size, len(ATTRIBUTES))
    )
    nn.init.xavier_uniform_(self.fc[0].weight)
    nn.init.xavier_uniform_(self.fc[-1].weight)

  def forward(self, input_ids, attention_mask):
    x = self.roBERTa(input_ids=input_ids, attention_mask=attention_mask)
    x = torch.mean(x.last_hidden_state, 1)
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
    pct_start=0.20,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100
  )
  '''
  scheduler = get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0.2 * num_epochs * len(train_loader),
      num_training_steps=num_epochs * len(train_loader)
  )
  '''

  # Early stopping parameters
  best_val_loss = float('inf')
  patience = 10
  patience_counter = 0
  completed_epochs = 0 # Track the number of completed epochs

  # Gradient clipping
  clipping_value = 0.1 #values to check: [0.01, 0.05, 0.1, 0.5, 1.0]

  # training loop
  for epoch in range(1, num_epochs + 1):
      model.train()  # put in training mode
      running_loss = 0.0
      epoch_time = time.time()
      for batch_data in train_loader:
          inputs = batch_data['input_ids'].to(device)
          attention_mask = batch_data['attention_mask'].to(device)
          labels = batch_data['labels'].to(device)
          # forward + backward + optimize
          outputs = model(inputs, attention_mask)  # forward pass
          loss = criterion(outputs, labels)  # calculate the loss
          optimizer.zero_grad()  # zero the parameter gradients
          loss.backward()  # backpropagation
          torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
          optimizer.step()  # update parameters
          scheduler.step()  # placed here because we are using OneCycleLR

          running_loss += loss.data.item()

      # Normalizing the loss by the total number of train batches
      running_loss /= len(train_loader)
      train_loss_per_epoch.append(running_loss)

      val_loss = Training.evaluate_model(model, val_loader, criterion, device)

      val_loss_per_epoch.append(val_loss)

      completed_epochs = epoch # Update completed_epochs

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

      # Calculate AUC for each attribute and then the average
      auc_scores = []
      for i in range(len(ATTRIBUTES)):
          try:
              auc = metrics.roc_auc_score(labels[:, i].astype(int), predictions[:, i])
              auc_scores.append(auc)
          except ValueError:
              # Handle cases where there's only one class in the labels
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


  print('==> Finished Training ...')

  # Plotting Loss
  plt.figure(figsize=(8, 5))
  # Use the actual number of completed epochs for the x-axis
  plt.plot(range(1, completed_epochs + 1), train_loss_per_epoch, label='Train Loss')
  plt.plot(range(1, completed_epochs + 1), val_loss_per_epoch, label='Val Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Curve')
  plt.legend()

  plt.tight_layout()
  plt.show()
  plt.savefig("Final.png")


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
    train_ds, val_ds, train_loader, val_loader, test_loader = Training.load_data(balanced_train_data, val_data, test_data, BATCH_SIZE, UCC_Dataset_BERT)

    train_model(model, train_loader, val_loader)

    labels = np.array(test_data[ATTRIBUTES])

    model_path = 'best_model.pth'
    Training.evaluate_saved_model(model_path, test_loader, test_data, "RoBERTa - Full Fine-Tuning\nROC-AUC Score on Test Set", True, UCC_classifier)
