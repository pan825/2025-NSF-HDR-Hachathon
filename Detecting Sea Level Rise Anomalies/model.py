import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from tqdm import tqdm

class Model(nn.Module):  # Change to inherit from nn.Module
    def __init__(self):
        """Initialize the model by loading the trained detector."""
        super().__init__()  # Call parent class initializer
        
        self.station_names = [
            'Atlantic_City', 'Baltimore', 'Eastport', 'Fort_Pulaski',
            'Lewes', 'New_London', 'Newport', 'Portland', 'Sandy_Hook',
            'Sewells_Point', 'The_Battery', 'Washington'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.future_data = {station: [] for station in self.station_names if station != "Newport"}
        self.current_position = 0
        self.load(os.path.join(os.path.dirname(__file__), "model.pt")) 

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

    def predict(self, X_test):
        predictions = []
        for station_name in self.station_names:
            prediction = self.predict_single_station(station_name, X_test)
            predictions.append(prediction)
        
        self.current_position += 1
        return np.array(predictions).reshape(1, -1)

    def predict_single_station(self, station_name, X_test):
        if station_name == "Newport":
            return 0

        coords = self.coords[station_name]
        current_data = self._extract_station_data(X_test, coords)
        
        X = self._prepare_window_data(station_name, current_data)        
        X_scaled = self.scalers[station_name].transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape)
        
        model = self.models[station_name]
        model.eval()
        with torch.no_grad():
            output = model(torch.FloatTensor(X_scaled).to(self.device))
            prediction = float(output.cpu().numpy())
            return 1 if prediction >= self.thresholds[station_name] else 0

    def _extract_station_data(self, X_test, coords):
        station_data = np.zeros(len(coords))
        for idx, (lat_idx, lon_idx) in enumerate(coords):
            station_data[idx] = X_test[0, lat_idx, lon_idx]
        return station_data

    def _prepare_window_data(self, station_name, current_data):
        self.future_data[station_name].append(current_data)
        current_position = self.current_position
        
        if current_position < self.window_size:
            window_data = np.array(self.future_data[station_name])
            if len(window_data) < self.window_size:
                repeats = self.window_size // len(window_data) + 1
                window_data = np.tile(window_data, (repeats, 1))[:self.window_size]
        else:
            start_idx = current_position - self.window_size + 1
            window_data = np.array(self.future_data[station_name][start_idx:current_position + 1])
        
        return window_data.T.reshape(1, len(current_data), -1)

    def load(self, path='model.pt'):
        model_data = torch.load(path)
        self.models = model_data['models']
        self.coords = model_data['coords']
        self.scalers = model_data['scalers']
        self.station_names = model_data['station_names']
        self.thresholds = model_data['thresholds']
        self.window_size = model_data['window_size']

        # Initialize future data buffer
        for station_name in self.station_names:
            if station_name != "Newport":
                self.future_data[station_name] = []

def get_sla_array(files):
    ds = xr.open_dataset(files)
    X = ds['sla'].values
    ds.close()
    return X

def get_prediction_data(input_dir='train_nc_folder'):
    train_data_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
    train_data_files.sort()
    
    print("Loading SLA arrays...")
    X_train_data = []
    for f in tqdm(train_data_files, desc="Loading data files"):
        X_train_data.append(get_sla_array(f))
    return X_train_data


if __name__ == "__main__":
    X_tests = get_prediction_data()
    m = Model()
    prediction_prob = [ m.predict(X_test) for X_test in X_tests ] 
    all_prediction = np.concatenate(prediction_prob, axis=0)
    print("Shape of output: " + str(all_prediction.shape))
