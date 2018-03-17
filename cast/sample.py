
import numpy as np

from . import image as im
from . import constants as const


def get_id_from_path(image_path):
    return image_path.stem.split('_')[0]



class Sample:

    def __init__(self, image_path):
        self.id = get_id_from_path(image_path)
        filename = image_path.stem
        if const.TRAINING in filename:
            self.type = const.TRAINING
        elif const.TEST in filename:
            self.type = const.TEST
        else:
            raise ValueError('Image type missing')

        self.dataset_dir = image_path.parents[1]
        self.filtered_dir = self.dataset_dir / 'filtered'
        self.rgb_path = image_path
        self.manual_1_path = self.dataset_dir / '1st_manual' \
                                              / f'{self.id}_manual1.gif'
        self.manual_2_path = self.dataset_dir / '2st_manual' \
                                              / f'{self.id}_manual2.gif'
        self.mask_path = self.dataset_dir / 'mask' \
                                          / f'{self.id}_{self.type}_mask.gif'
        self.prediction_path = self.dataset_dir / 'predicted' \
                                                / f'{self.id}_predicted.png'
        self.frangi_path = self.filtered_dir / f'{self.id}_frangi.tiff'

        self._rgb_data = None
        self._frangi_data = None
        self._mask = None
        self._labels = None
        self.prediction = None


    @property
    def rgb_data(self):
        if self._rgb_data is None:
            self._rgb_data = im.read(self.rgb_path)
        return self._rgb_data


    @property
    def mask(self):
        if self._mask is None:
            self._mask = im.read(self.mask_path).ravel() > 0
            self._mask = im.erode(self._mask, 2)
        return self._mask


    @property
    def labels(self):
        if self._labels is None:
            self._labels = im.read(self.manual_1_path).ravel()
        return self._labels


    @property
    def mask_indices(self):
        return np.where(self.mask)[0]


    @property
    def shape(self):
        return self.rgb_data.shape[:-1]


    @property
    def N(self):
        si, sj = self.shape
        return si * sj


    @property
    def X(self):
        features = self.get_features()
        X = features[self.mask_indices, :]
        return X


    @property
    def y(self):
        y = self.labels[self.mask_indices]
        return y


    @property
    def frangi_data(self):

        if not self.frangi_path.is_file():
            self._frangi_data = im.frangi_filter(self.rgb_data)
            im.write(self._frangi_data, self.frangi_path)
        if self._frangi_data is None:
            self._frangi_data = im.read(self.frangi_path)
        return self._frangi_data


    def get_features(self):
        rgb_features = self.rgb_data.reshape(self.N, 3)
        frangi_features = self.frangi_data.reshape(self.N, 1)
        features = np.hstack([rgb_features, frangi_features])
        return features


    def save_prediction(self, y_predicted, closing=True):
        prediction = np.zeros(self.N, np.uint8)
        prediction[self.mask_indices] = y_predicted
        self.prediction = prediction.reshape(*self.shape)
        if closing:
            self.prediction = im.closing(self.prediction)
        im.write(self.prediction, self.prediction_path)
