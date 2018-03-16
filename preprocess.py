from time import time
from pathlib import Path

import numpy as np
from skimage import io, color
from sklearn.ensemble import ExtraTreesClassifier

HEIGHT, WIDTH = 584, 565


def ensure_dir(path):
    """Make sure that the directory and its parents exists"""
    path = Path(path)
    if path.exists():
        return
    is_dir = not path.suffixes
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


class DataSet:

    def __init__(self, data_dir):
        self.dir = data_dir
        self.images_dir = self.dir / 'images'
        self.mask_dir = self.dir / 'mask'
        self.manual_dir = self.dir / '1st_manual'
        self.filtered_dir = self.dir / 'filtered'
        self.features_dir = self.dir / 'features'
        self.frangi_dir = self.filtered_dir / 'frangi'

        self.rgb_features_path = self.features_dir / 'rgb.npy'
        self.mask_vector_path = self.features_dir / 'mask.npy'
        self.labels_vector_path = self.features_dir / 'labels.npy'


    def get_images_paths(self):
        return self.images_dir.glob('*.tif')


    def get_rgb_features(self):
        if self.rgb_features_path.exists():
            rgb = np.load(str(self.rgb_features_path))
        else:
            rgb = self.generate_rgb_features()
        return rgb


    def generate_rgb_features(self, force=False):
        if self.rgb_features_path.exists() and not force:
            print(self.rgb_features_path.name, 'already exists')
            return
        paths = self.get_images_paths()
        num_images = len(list(paths))
        num_pixels_one = HEIGHT * WIDTH
        num_pixels_total = num_pixels_one * num_images
        rgb = np.empty((num_pixels_total, 3), np.uint8)
        for i, path in enumerate(paths):
            print(path.name)
            image = io.imread(str(path))  # H x W x 3
            idx_ini = i * num_pixels_one
            idx_fin = (i + 1) * num_pixels_one
            rgb[idx_ini : idx_fin, :] = image.reshape(num_pixels_one, c)
        ensure_dir(self.rgb_features_path)
        np.save(str(self.rgb_features_path), rgb)
        return rgb


    def get_masks_paths(self):
        return self.mask_dir.glob('*.gif')


    def get_masks_vector(self):
        if self.mask_vector_path.exists():
            masks_vector = np.load(str(self.mask_vector_path))
        else:
            masks_vector = self.generate_mask_vector()
        return masks_vector


    def generate_mask_vector(self, force=False):
        if self.mask_vector_path.exists() and not force:
            print(self.mask_vector_path.name, 'already exists')
            return
        paths = list(self.get_masks_paths())
        num_images = len(paths)
        num_pixels_one = HEIGHT * WIDTH
        num_pixels_total = num_pixels_one * num_images
        masks_vector = np.empty(num_pixels_total, bool)

        for i, path in enumerate(paths):
            print(path.name)
            image = io.imread(str(path)) > 0  # H x W
            idx_ini = i * num_pixels_one
            idx_fin = (i + 1) * num_pixels_one
            masks_vector[idx_ini : idx_fin] = image.reshape(num_pixels_one)
        ensure_dir(self.mask_vector_path)
        np.save(str(self.mask_vector_path), masks_vector)
        return masks_vector


    def get_labels_paths(self):
        return self.manual_dir.glob('*.gif')


    def get_labels_vector(self):
        if self.labels_vector_path.exists():
            labels_vector = np.load(str(self.labels_vector_path))
        else:
            labels_vector = self.generate_labels_vector()
        return labels_vector


    def generate_labels_vector(self):
        if self.labels_vector_path.exists() and not force:
            print(self.labels_vector_path.name, 'already exists')
            return
        paths = list(self.get_labels_paths())
        num_images = len(paths)
        num_pixels_one = HEIGHT * WIDTH
        num_pixels_total = num_pixels_one * num_images
        labels_vector = np.empty(num_pixels_total, bool)

        for i, path in enumerate(paths):
            print(path.name)
            image = io.imread(str(path)) > 0  # H x W
            idx_ini = i * num_pixels_one
            idx_fin = (i + 1) * num_pixels_one
            labels_vector[idx_ini : idx_fin] = image.reshape(num_pixels_one)
        ensure_dir(self.labels_vector_path)
        np.save(str(self.labels_vector_path), labels_vector)
        return labels_vector


    def make_frangi_images(self):
        for path in self.get_images_paths():
            return


    def get_all_features(self):
        rgb = self.get_rgb_features()
        all_features = rgb
        return all_features


    def get_all(self):
        features = self.get_all_features()
        labels = self.get_labels_vector()
        mask = self.get_masks_vector()

        return features, labels, mask



class Image:

    def __init__(self, image_path):
        self.id = image_id
        self.image_data = None




def main():
    return
dataset_dir = Path(Path.home(), 'Desktop', 'DRIVE')
test_dir = dataset_dir / 'test'
training_dir = dataset_dir / 'training'

test_set = DataSet(test_dir)
training_set = DataSet(training_dir)
X_training, y_training = training_set.get_masked_features_array()
X_test, y_test = training_set.get_masked_features_array()
clf = ExtraTreesClassifier()

start = time()
print('Training...')
clf.fit(X_training, y_training)
print('Training time:', time() - start, 'seconds')

start = time()
print('Testing...')
clf = ExtraTreesClassifier()
clf.fit(np.random.rand(5,3), [0, 0, 0, 0, 0])
score = clf.score(X_test, y_test)
print('Testing time:', time() - start, 'seconds')

print('Score:', score)




if __name__ == '__main__':
    main()
