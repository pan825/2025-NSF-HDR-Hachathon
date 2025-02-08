import os
print(os.getcwd())
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/fine_tune/Hackathon-Butterfly-Hybrid-Detection-Main/DINO_notebook/')

import torch
import sys
import os
import csv
from pathlib import Path
# Add your path of library
sys.path.append('../DINO_train')
import training

training.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
model = training.get_dino_model(dino_name='facebook/dinov2-base').to(training.DEVICE)
training.BATCH_SIZE = 4

# Location to save checkpoints and results
training.CLF_SAVE_DIR = Path('./trained_clfs')
os.makedirs(training.CLF_SAVE_DIR, exist_ok=True)
print(training.DEVICE)
# print(model)
print(type(model))

img_path = "/content/drive/MyDrive/CWL_butterfly/Hackathon-Butterfly-Hybrid-Detection-main/Hackathon-Butterfly-Hybrid-Detection-main/input_data/New_both2"
data_path = "/content/drive/MyDrive/CWL_butterfly/Hackathon-Butterfly-Hybrid-Detection-main/Hackathon-Butterfly-Hybrid-Detection-main/input_data/0130_final.csv"

train_data, test_data = training.load_data(data_path, img_path, test_size=0.2)

# Create dataloader
tr_sig_dloader, test_dl = training.prepare_data_loaders(train_data, test_data, img_path)
print(training.data_transforms())

# Extract visual features from model
try:
    tr_features, tr_labels, test_features, test_labels = training.extract_features(tr_sig_dloader, test_dl, model)
    print(tr_features)
except Exception as e:
    print("Error in extracting features")
    print(e)

# Train classifier with visual features
csv_output, score_output = training.train_and_evaluate(tr_features, tr_labels, test_features, test_labels)

# Save evaluation results
csv_filename = training.CLF_SAVE_DIR / "classifier_evaluation_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Configuration", "AUC", "Precision", "Recall", "F1-score"])
    writer.writerows(csv_output)

# Save individual scores
scores_filename = training.CLF_SAVE_DIR / "classifier_scores.csv"
with open(scores_filename, mode='w', newline='') as score_file:
    score_writer = csv.writer(score_file)
    score_writer.writerow(["Index", "Score", "True Label"])
    score_writer.writerows(score_output)
