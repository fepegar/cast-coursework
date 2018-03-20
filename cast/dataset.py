from pathlib import Path

import numpy as np

from .sample import Sample

class DataSet:

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.images_dir = self.dir / 'images'
        self.mask_dir = self.dir / 'mask'
        self.manual_dir = self.dir / '1st_manual'
        self.filtered_dir = self.dir / 'filtered'
        self.features_dir = self.dir / 'features'
        self.frangi_dir = self.filtered_dir / 'frangi'

        self.rgb_features_path = self.features_dir / 'rgb.npy'
        self.mask_vector_path = self.features_dir / 'mask.npy'
        self.labels_vector_path = self.features_dir / 'labels.npy'

        self.rgb_paths = sorted(list(self.images_dir.glob('*.tif')))
        self.samples = [Sample(path) for path in self.rgb_paths]


    @property
    def X(self):
        X = np.vstack([sample.X for sample in self.samples])
        return X


    @property
    def y(self):
        y = np.hstack([sample.y for sample in self.samples])
        return y


    def save_predictions(self, y_predicted):
        idx_ini = 0
        for sample in self.samples:
            idx_fin = idx_ini + len(sample.mask_indices)
            y_predicted_sample = y_predicted[idx_ini : idx_fin]
            sample.save_prediction(y_predicted_sample)
            idx_ini = idx_fin


    def get_samples_class_imbalance(self):
        percentages = [s.get_background_percentage() for s in self.samples]
        return np.array(percentages)
