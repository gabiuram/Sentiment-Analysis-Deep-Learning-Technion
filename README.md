# Sentiment-Analysis-Deep-Learning-Technion
Repository for project of course Deep Learning (00460217) from Technion

## Repository description:
The repository contains 4 folders of interest:\

* ```bin```: contains the code for training each of the models
* ```data```: contains the data. Reference from [GitHub repo](https://github.com/conversationai/unhealthy-conversations)
* ```Report Images```: Images generated with data from the models
* ```src```: Code with basic functions used across multiple models

## Set up the environment locally:
We use poetry as our package manager, so run the following commands:
1. ```pip install poetry```
2. ```poetry lock```
3. ```poetry install```
And this will install all dependencies

## Set up the environment o Google Collab:
1. ```!git clone https://github.com/gabiuram/Sentiment-Analysis-Deep-Learning-Technion.git```
2. ```%cd Sentiment-Analysis-Deep-Learning-Technion```
3. ```!pip install poetry```
4. ```!poetry config virtualenvs.create false```
5. ```!poetry install```

## Presentation:
Check our presentation on [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1)

## Hyperparameters:

### DistilRoBERTa:

| **Hyperparameter**               | **Value**         |
|-----------------------------------|-------------------|
| Batch Size                        | 256               |
| Number of Epochs                  | 21                |
| Learning Rate                     | 3 × 10⁻⁵          |
| Weight Decay                      | 0.001             |
| Optimizer                         | AdamW             |
| Learning Rate Scheduler           | OneCycleLR        |
| Gradient Clipping Value           | 0.1               |
| Dropout Rate                      | 0.4               |
| Maximum Token Length              | 128               |
| Additional Healthy Samples*       | 5000              |


### BiLSTM:

| **Hyperparameter**               | **Value**         |
|-----------------------------------|-------------------|
| Batch Size                        | 16                |
| Number of Epochs                  | 21                |
| Learning Rate                     | 5 × 10⁻⁵          |
| Weight Decay                      | 0.01              |
| Optimizer                         | AdamW             |
| Learning Rate Scheduler           | OneCycleLR        |
| Gradient Clipping Value           | 0.1               |
| Dropout Rate                      | 0.3               |
| Maximum Token Length              | 128               |
| Additional Healthy Samples*       | 11500             |
| Embedding Dimension               | 200               |
| Tokenizer Vocab Size              | 10000             |
| Convolution Kernel Size           | 3                 |
| Convolution Padding               | 1                 |
| BiLSTM Num Layers                 | 1                 |


### RoBERTa Large:

| **Hyperparameter**                | **Value**                    |
|------------------------------------|------------------------------|
| LoRA Rank (r)                      | 8                            |
| LoRA Alpha                         | 16                           |
| LoRA Dropout                       | 0.05                         |
| Batch Size                         | 64                           |
| Number of Epochs                   | 21                           |
| Learning Rate                      | 5 × 10⁻⁵                     |
| Weight Decay                       | 0.001                        |
| Optimizer                          | AdamW                        |
| Learning Rate Scheduler            | OneCycleLR                   |
| Dropout Rate (Classifier)          | 0.1                          |
| Maximum Token Length               | 64 (due to memory issues)    |
| Additional Healthy Samples*        | 5000                         |
| Gradient Clipping Value            | 0.1                          |





