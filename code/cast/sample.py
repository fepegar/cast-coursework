
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
        self.predicted_dir = self.dataset_dir / 'predicted'
        self.rgb_path = image_path

        self.manual_1_path = self.dataset_dir / '1st_manual' \
                                              / f'{self.id}_manual1.gif'
        self.manual_2_path = self.dataset_dir / '2nd_manual' \
                                              / f'{self.id}_manual2.gif'
        self.mask_path = self.dataset_dir / 'mask' \
                                          / f'{self.id}_{self.type}_mask.gif'

        # Predictions
        self.prediction_prob_path = self.predicted_dir \
                                    / f'{self.id}_predicted_prob.png'
        self.prediction_prob_rgb_path = self.predicted_dir \
                                    / f'{self.id}_predicted_prob_rgb.png'
        self.prediction_mask_path = self.predicted_dir \
                                    / f'{self.id}_predicted_mask.png'
        self.driu_path = self.dataset_dir / 'driu' \
                                              / f'{self.id}_test.png'

        # Filters
        self.frangi_path = self.filtered_dir / f'{self.id}_frangi.tiff'
        self.sobel_path = self.filtered_dir / f'{self.id}_sobel.tiff'
        self.laplacian_path = self.filtered_dir / f'{self.id}_laplacian.tiff'
        self.lab_path = self.filtered_dir / f'{self.id}_lab.tiff'
        self.hsv_path = self.filtered_dir / f'{self.id}_hsv.tiff'

        self._rgb_data = None
        self._hsv_data = None
        self._lab_data = None
        self._laplacian_data = None
        self._frangi_data = None
        self._sobel_data = None
        self._mask = None
        self._labels = None
        self.prediction = None


    @property
    def rgb_data(self):
        if self._rgb_data is None:
            self._rgb_data = im.read(self.rgb_path)
        return self._rgb_data


    @property
    def lab_data(self):
        if not self.lab_path.is_file():
            self._lab_data = im.lab_filter(self.rgb_data)
            im.write(self._lab_data, self.lab_path)
        if self._lab_data is None:
            self._lab_data = im.read(self.lab_path)
        return self._lab_data


    @property
    def hsv_data(self):
        if not self.hsv_path.is_file():
            self._hsv_data = im.hsv_filter(self.rgb_data)
            im.write(self._hsv_data, self.hsv_path)
        if self._hsv_data is None:
            self._hsv_data = im.read(self.hsv_path)
        return self._hsv_data


    @property
    def mask(self):
        if self._mask is None:
            self._mask = im.read(self.mask_path)
            self._mask = im.erode(self._mask, 2) > 0
        return self._mask


    @property
    def labels(self):
        """Return the ground truth image"""
        if self._labels is None:
            self._labels = im.read(self.manual_1_path)
        return self._labels


    @property
    def mask_indices(self):
        return np.where(self.mask.ravel())[0]


    @property
    def shape(self):
        return self.rgb_data.shape[:-1]


    @property
    def N(self):
        si, sj = self.shape
        return si * sj


    @property
    def X(self):
        """Return the masked features array"""
        features = self.get_features()
        X = features[self.mask_indices, :]
        return X


    @property
    def y(self):
        """Return a vector with the masked ground truth labels"""
        y = self.labels.ravel()[self.mask_indices]
        return y


    @property
    def frangi_data(self):
        if not self.frangi_path.is_file():
            self._frangi_data = im.frangi_filter(self.rgb_data)
            im.write(self._frangi_data, self.frangi_path)
        if self._frangi_data is None:
            self._frangi_data = im.read(self.frangi_path)
        return self._frangi_data


    @property
    def sobel_data(self):
        if not self.sobel_path.is_file():
            self._sobel_data = im.sobel_filter(self.rgb_data)
            im.write(self._sobel_data, self.sobel_path)
        if self._sobel_data is None:
            self._sobel_data = im.read(self.sobel_path)
        return self._sobel_data


    @property
    def laplacian_data(self):
        if not self.laplacian_path.is_file():
            self._laplacian_data = im.laplacian_filter(self.rgb_data)
            im.write(self._laplacian_data, self.laplacian_path)
        if self._laplacian_data is None:
            self._laplacian_data = im.read(self.laplacian_path)
        return self._laplacian_data


    def get_confusion_map(self):
        binary_labels = self.labels
        binary_prediction = im.read(self.prediction_mask_path)
        confusion_map = im.get_confusion_map(binary_labels, binary_prediction)
        return confusion_map


    def get_driu_confusion_map(self):
        binary_labels = self.labels
        driu_prediction = im.read(self.driu_path).astype(float)
        driu_prediction /= driu_prediction.max()
        binary_prediction = driu_prediction > 0.5
        confusion_map = im.get_confusion_map(binary_labels, binary_prediction)
        return confusion_map


    def dice_score(self, confusion_map=None):
        if confusion_map is None:
            confusion_map = self.get_confusion_map()
        TP = confusion_map['TP']
        FP = confusion_map['FP']
        FN = confusion_map['FN']
        dice_score = 2 * TP / (2 * TP + FP + FN)
        return dice_score


    def get_precision_recall(self):
        confusion_map = self.get_confusion_map()
        TP = confusion_map['TP']
        FP = confusion_map['FP']
        FN = confusion_map['FN']
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return precision, recall


    def get_background_percentage(self):
        bg_pixels = np.count_nonzero(self.y == 0)
        return bg_pixels / self.y.size


    def get_features(self):
        # Color
        rgb_features = self.rgb_data.reshape(self.N, 3)
        lab_features = self.lab_data.reshape(self.N, 3)
        hsv_features = self.hsv_data.reshape(self.N, 3)

        # Edges and vesselness
        sobel_features = self.sobel_data.reshape(self.N, 1)
        frangi_features = self.frangi_data.reshape(self.N, 1)
        laplacian_features = self.laplacian_data.reshape(self.N, 1)
        features = [
            rgb_features,
            lab_features,
            hsv_features,
            sobel_features,
            frangi_features,
            laplacian_features,
        ]
        features = np.hstack(features)
        return features


    def save_prediction(self, y_predicted, filter_result=False):
        prediction = np.zeros(self.N)
        prediction[self.mask_indices] = y_predicted
        self.prediction = prediction.reshape(*self.shape)
        if filter_result:
            self.prediction = im.median_filter(self.prediction)
        prediction_rgb = im.grey2rgb(self.prediction, self.mask)
        im.write(prediction_rgb, self.prediction_prob_rgb_path)
        im.write(self.prediction, self.prediction_prob_path)
        im.write(im.img_as_uint(self.prediction > 0.5),
                 self.prediction_mask_path)
