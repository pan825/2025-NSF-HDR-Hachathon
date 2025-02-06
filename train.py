import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # SMOTE for oversampling
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.cluster import KMeans

# ----------------------- New: Focal Loss -----------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Args:
        alpha (float): 平衡正負樣本的係數 (預設為 1)
        gamma (float): 調整難易樣本權重的參數 (預設為 2)
        reduction (str): 損失彙總方式 ('mean', 'sum' 或 'none')
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
# ---------------------------------------------------------------

class Model(nn.Module):
    def __init__(self):
        """Initialize the model by loading the trained detector."""
        super().__init__()  # Call parent class initializer
        
        self.station_names = [
            'Atlantic_City', 'Baltimore', 'Eastport', 'Fort_Pulaski',
            'Lewes', 'New_London', 'Newport', 'Portland', 'Sandy_Hook',
            'Sewells_Point', 'The_Battery', 'Washington'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.coords = {}
        self.scalers = {}
        self.thresholds = {}
        self.class_weights = {}
        self.window_size = 60  # =k_days

    def forward(self, x, hidden=None):
        """
        Forward pass for the LSTM Anomaly Detector neural network.
        
        Args:
            x (torch.Tensor): Input tensor
            hidden (tuple, optional): Hidden state of the LSTM
        
        Returns:
            torch.Tensor: Anomaly detection probability
        """
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        context = context.squeeze(1)
        output = self.classifier(context)
        return output

    def initialize_model(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM Anomaly Detector neural network architecture.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int, optional): Number of hidden units in LSTM. Defaults to 64.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def get_sla_array(self, files):
        ds = xr.open_dataset(files)
        X = ds['sla'].values
        ds.close()
        return X

    def get_prediction_data(self, input_dir='train_nc_folder'):
        train_data_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
        train_data_files.sort()
        
        print("Loading SLA arrays...")
        X_train_data = []
        for f in tqdm(train_data_files, desc="Loading data files"):
            X_train_data.append(self.get_sla_array(f))
        return X_train_data

    def time_series_dataset(self, features, labels):
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(labels)
        )

    def calculate_class_weights(self, labels):
        positive = np.sum(labels == 1)
        negative = np.sum(labels == 0)
        weight_positive = negative / (positive + negative)
        weight_negative = positive / (positive + negative)
        return torch.FloatTensor([weight_negative, weight_positive]).to(self.device)

    def calculate_metrics(self, true_labels, predictions, threshold=0.5):
        binary_preds = (predictions >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        f1 = f1_score(true_labels, binary_preds)
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
        
        return {
            'f1': f1,
            'tpr': tpr[1],
            'fpr': fpr[1],
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }

    def fit(self, csv_path, data_list, k_days=7, n_points=200):
        for station_name in self.station_names:
            if station_name == "Newport":  # Skip training for Newport
                print(f"Skipping training for {station_name} due to zero anomalies.")
                continue

            print(f"\nTraining model for station: {station_name}")
            
            # Find valid coordinates
            coords = self.find_valid_coords(data_list, n_points)
            if coords is None:
                print(f"Warning: No valid coordinates found for {station_name}")
                continue
            self.coords[station_name] = coords
            
            # Prepare station data
            df = pd.read_csv(csv_path, parse_dates=['t'])
            station_data = df[['t', station_name]].copy()
            station_data = station_data[station_data['t'] <= '2013-12-31']
            
            dates = station_data['t'].dt.strftime('%Y%m%d').tolist()
            labels = station_data[station_name].values
            
            # Extract window data
            window_data, valid_indices = self.extract_window_data(
                data_list, coords, dates, k_days, labels=labels
            )
            
            if not window_data:
                print(f"Warning: No valid windows found for {station_name}")
                continue
                
            X3d = np.array(window_data)
            y = labels[valid_indices]
            
            # Train station-specific model
            model = self.train_station_model(station_name, X3d, y)
            self.models[station_name] = model
            
            print(f"Model trained successfully for {station_name}")

    def find_valid_coords(self, data_list, n_points=200, selection_method='xgboost'):
        """
        Advanced method for selecting coordinates for training.
        
        Args:
            data_list (list): List of 3D SLA arrays
            n_points (int): Number of coordinates to select
            selection_method (str): Method for coordinate selection ('random', 'variance', 
                                'extreme_events', 'spatial_correlation', 'xgboost')
        
        Returns:
            np.ndarray: Selected coordinates
        """
        # Combine data across all time steps
        combined_data = np.concatenate(data_list, axis=0)
        
        combined_valid_mask = None
        for arr in tqdm(data_list, desc="Finding valid coordinates"):
            arr_2d = arr[0, :, :]
            valid_mask = ~np.isnan(arr_2d)
            if combined_valid_mask is None:
                combined_valid_mask = valid_mask
            else:
                combined_valid_mask &= valid_mask
        
        coords_all = np.argwhere(combined_valid_mask)
        if len(coords_all) == 0:
            return None
        
        # Compute coordinate scores based on selection method
        if selection_method == 'variance':
            coord_variances = np.nanvar(combined_data, axis=0)[combined_valid_mask]
            variance_ranking = np.argsort(coord_variances)[::-1]
            selected_coords = coords_all[variance_ranking[:n_points]]
        
        elif selection_method == 'extreme_events':
            threshold_percentiles = [1, 99]
            extreme_counts = np.zeros(len(coords_all))
            
            for percentile in threshold_percentiles:
                lower_threshold = np.nanpercentile(combined_data, percentile)
                upper_threshold = np.nanpercentile(combined_data, 100 - percentile)
                
                extreme_mask = (
                    (combined_data <= lower_threshold) | 
                    (combined_data >= upper_threshold)
                )
                extreme_counts += np.sum(extreme_mask, axis=0)[combined_valid_mask]
            
            extreme_ranking = np.argsort(extreme_counts)[::-1]
            selected_coords = coords_all[extreme_ranking[:n_points]]
        
        elif selection_method == 'spatial_correlation':
            from scipy.spatial import distance
            
            def compute_spatial_score(coord_index, coords_data):
                coord = coords_data[coord_index]
                spatial_distances = distance.cdist([coord], coords_data)[0]
                nearby_mask = spatial_distances < 0.5
                nearby_coords_indices = np.where(nearby_mask)[0]
                
                if len(nearby_coords_indices) == 0:
                    return 0
                
                nearby_data = combined_data[:, tuple(coords_data[nearby_coords_indices].T)]
                
                try:
                    spatial_variance = np.nanvar(nearby_data, axis=1)
                    return np.mean(spatial_variance)
                except Exception as e:
                    print(f"Error in spatial variance computation: {e}")
                    return 0
            
            spatial_scores = np.array([
                compute_spatial_score(i, coords_all) 
                for i in range(len(coords_all))
            ])
            
            spatial_ranking = np.argsort(spatial_scores)[::-1]
            selected_coords = coords_all[spatial_ranking[:n_points]]
        
        elif selection_method == 'xgboost':   
            print(f"Total valid coordinates: {len(coords_all)}")
            print(f"Selecting {n_points} training points...")
                
            # 1. Prepare features (coordinates) and target (SLA values)
            X = coords_all.astype(float)
            y = np.array([np.nanmean(combined_data[:, coord[0], coord[1]]) for coord in coords_all])
                
            # 2. Train XGBoost to identify important regions
            xgb_model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42)
            
            xgb_model.fit(X, y)
                
            # 3. Get predictions and sort by prediction confidence
            predictions = xgb_model.predict(X)
            importance_scores = np.abs(predictions - y)  # Use prediction error as importance
                
            # 4. Select top 2N points based on importance
            n_candidates = min(len(coords_all), n_points * 2)
            top_indices = np.argsort(importance_scores)[-n_candidates:]
            candidate_coords = coords_all[top_indices]
                
            # 5. Use k-means++ to select diverse points from candidates
            kmeans = KMeans(
                n_clusters=n_points,
                init='k-means++',
                n_init=10,
                random_state=42)
                
            scaler = StandardScaler()
            candidate_coords_scaled = scaler.fit_transform(candidate_coords)
                
            kmeans.fit(candidate_coords_scaled)
                
            selected_coords = np.array([
                candidate_coords[
                    np.argmin(np.linalg.norm(candidate_coords_scaled - center, axis=1))
                ] for center in kmeans.cluster_centers_
            ])
                
            print(f"Selected {len(selected_coords)} training points")
        
        else:
            np.random.seed(42)
            np.random.shuffle(coords_all)
            selected_coords = coords_all[:n_points]

        return selected_coords

    def extract_window_data(self, data_list, coords, dates, k_days, labels=None):
        window_data = []
        valid_indices = []
        
        for i in tqdm(range(len(dates)), desc="Extracting window data"):
            if labels is not None and np.isnan(labels[i]):
                continue
                
            if i < k_days:
                indices = range(0, k_days)
            else:
                indices = range(i - k_days + 1, i + 1)
            
            data_array = np.full((coords.shape[0], k_days), np.nan)
            valid_window = True
            
            for j, idx in enumerate(indices):
                if idx >= len(data_list):
                    valid_window = False
                    break
                    
                sla_3d = data_list[idx]
                for p_idx, (lat_idx, lon_idx) in enumerate(coords):
                    value = sla_3d[0, lat_idx, lon_idx]
                    if np.isnan(value):
                        valid_window = False
                        break
                    data_array[p_idx, j] = value
                
                if not valid_window:
                    break
            
            if valid_window:
                window_data.append(data_array)
                valid_indices.append(i)
        
        return window_data, valid_indices

    def train_station_model(self, station_name, train_X3d, train_y, batch_size=32, epochs=100):
        # Split data into train and validation sets
        print(f"Coordinate selection statistics for {station_name}:")
        print(f"Total training windows: {len(train_X3d)}")
        print(f"Anomaly windows: {np.sum(train_y == 1)}")
        print(f"Normal windows: {np.sum(train_y == 0)}")

        X_train, X_val, y_train, y_val = train_test_split(
            train_X3d, train_y, test_size=0.2, random_state=42
        )

        # ----------------- SMOTE with adjusted ratio -----------------
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        smote = SMOTE(random_state=42, sampling_strategy=1/2)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_2d, y_train)
        X_train_resampled = X_train_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
        # -------------------------------------------------------------

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train_resampled.reshape(-1, X_train_resampled.shape[-1])
        ).reshape(X_train_resampled.shape)
        X_val_scaled = scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)

        self.scalers[station_name] = scaler

        class_weights = self.calculate_class_weights(y_train_resampled)
        self.class_weights[station_name] = class_weights

        # ----------------- Create datasets and DataLoaders -----------------
        train_features = torch.FloatTensor(X_train_scaled)
        train_labels = torch.FloatTensor(y_train_resampled)
        train_dataset = list(zip(train_features, train_labels))
        
        y_np = y_train_resampled  # numpy array
        class_sample_count = np.array([np.sum(y_np == t) for t in np.unique(y_np)])
        weight_per_class = 1. / class_sample_count
        samples_weight = np.array([weight_per_class[int(t)] for t in y_np])
        samples_weight = torch.from_numpy(samples_weight).float()
        
        sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(samples_weight),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
        val_dataset = list(zip(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # --------------------------------------------------------------------

        self.initialize_model(
            input_size=X_train.shape[2],
            hidden_size=512,
            num_layers=3
        )
        
        # Move model to device
        model = self.to(self.device)
        
        # ----------------- Loss and Optimization -----------------
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        # -----------------------------------------------------------

        best_f1 = 0
        best_model_state = None
        patience = 30
        patience_counter = 0

        # Training loop
        pbar = tqdm(range(epochs), desc=f"Training {station_name}")
        for epoch in pbar:
            model.train()
            train_predictions = []
            train_labels_list = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)

                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_labels_list.extend(batch_y.cpu().numpy())
            
            # Validation phase
            model.eval()
            val_predictions = []
            val_labels_list = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = self.forward(batch_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_labels_list.extend(batch_y.numpy())
            
            # Convert predictions/labels to numpy arrays
            train_predictions = np.array(train_predictions)
            train_labels_arr = np.array(train_labels_list)
            val_predictions = np.array(val_predictions)
            val_labels_arr = np.array(val_labels_list)
            
            # Find optimal threshold based on ROC curve
            fpr, tpr, thresholds = roc_curve(val_labels_arr, val_predictions)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            train_metrics = self.calculate_metrics(train_labels_arr, train_predictions, optimal_threshold)
            val_metrics = self.calculate_metrics(val_labels_arr, val_predictions, optimal_threshold)
            
            scheduler.step(val_metrics['f1'])
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_model_state = model.state_dict().copy()
                self.thresholds[station_name] = optimal_threshold
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                pbar.write(f"Early stopping at epoch {epoch}")
                break
            
            pbar.set_postfix({
                'Train F1': f"{train_metrics['f1']:.4f}",
                'Val F1': f"{val_metrics['f1']:.4f}"
            })

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model.eval()

    def save(self, path):
        model_data = {
            'models': self.models,
            'coords': self.coords,
            'scalers': self.scalers,
            'station_names': self.station_names,
            'thresholds': self.thresholds,
            'window_size': self.window_size
        }
        torch.save(model_data, path)

def train_model():
    model = Model()
    
    print("Loading training data...")
    train_data = model.get_prediction_data('train_nc_folder')
    
    print("\nStarting model training...")
    model.fit(
        csv_path='train.csv',
        data_list=train_data,
        k_days=60,
        n_points=50  # FIXED 50 POINTS
    )
    
    print("\nSaving model...")
    model.save('model.pt')
    print("Model saved successfully!")

# Main execution block
if __name__ == "__main__":
    train_model()


