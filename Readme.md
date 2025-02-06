# Procedure to Run the Code

**Reminder:** Paths in the code may need to be modified according to user specifications.

## 1. Prepare Training Datasets
- Training Datasets required:
  - **Copernicus_ENA_Satellite_Maps_Training_Data**
    - Create a folder named `train_nc_folder` to store the `.nc` files.
    - We only use satellite maps data from 1997 to 2013.
  - **Training_Anomalies_Station_Data**
    - Create a folder named `Training_Anomalies_Station_Data` to store the `.csv` files.
- Download the training datasets from the following links:
  - [Copernicus_ENA_Satellite_Maps_Training_Data](https://drive.google.com/drive/folders/14ylyfXXiBunScZiVbdzhYxh2bTdhfVxO)
  - [Training_Anomalies_Station_Data](https://drive.google.com/drive/folders/1Uawcu_ocPE69Mx0nokOQI56KZ-l7pjbz)

## 2. Install Required Libraries
- Install the dependencies listed in `training_requirements.txt` to run `train.py`.
- **Notice that you only need to install the dependencies listed in `requirements.txt` to run `model.py`.**

## 3. Generate Truncated Training Data
- Run `Create_DATA.py` to produce a truncated training CSV file that records anomalies for each station from 1997 to 2013.

## 4. Train the LSTM Anomaly Detection Model
- Run `train.py` to train the LSTM model for predicting anomalies at each stations .
- The trained model weights will be stored under the `model.pt`.

## 5. Infer Final Predictions
- Run `model.py` to infer the final predictions of anomalies for the 12 stations from 2014 to 2023, notice that model weight path may need to be adjusted.

# Summary Description of the Model
- **The model is separated into two parts:**

  1. **Data Processing and Feature Engineering**
     - **Data Loading:**
       - Reads Sea Level Anomaly (SLA) data from NetCDF files using `xarray`.
       - Loads a CSV file containing station labels and timestamps.
     - **Coordinate Selection:**
       - Identifies valid data points using multiple methods:
         - An XGBoost regression approach combined with k-means clustering is used.
     - **Window Extraction:**
       - Extracts sliding window data (over a configurable number of days, `k_days`) from the SLA arrays.
       - Ensures that all selected coordinates in each window have valid (non-NaN) values.
     - **Handling Data Imbalance:**
       - Applies SMOTE to oversample the minority (anomaly) class.
       - Uses a Weighted Random Sampler during training to balance batch sampling.
     - **Feature Scaling:**
       - Standardizes the input features using `StandardScaler`.

  2. **Model Architecture, Training, and Inference**
     - **Model Architecture:**
       - **LSTM with Attention:**
         - A bidirectional LSTM processes the sequential data.
         - An attention mechanism computes weights over the LSTM outputs to generate a context vector.
       - **Classifier:**
         - A feedforward network that takes the context vector and outputs an anomaly probability using a Sigmoid activation.
       - **Focal Loss:**
         - Implements a custom Focal Loss function to focus training on harder-to-classify examples and to address class imbalance.
     - **Training Process:**
       - Splits data into training and validation sets.
       - Applies SMOTE and feature scaling to the training data.
       - Uses AdamW optimizer with gradient clipping and a learning rate scheduler (ReduceLROnPlateau).
       - Monitors performance using F1 score and adjusts the classification threshold based on ROC curve analysis.
       - Implements early stopping to avoid overfitting.
     - **Per-Station Modeling:**
       - Trains separate models for each station (with specific handling for known issues, e.g., skipping "Newport").
       - Stores station-specific information such as selected coordinates, scalers, and optimal thresholds.
     - **Model Saving and Inference:**
       - Saves the complete model state (including architecture, trained weights, and metadata) for future inference.