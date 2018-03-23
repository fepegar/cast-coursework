#!/usr/bin/env python3

from pathlib import Path

from cast import DataSet
from cast import image as im

dataset_dir = Path(Path.home(), 'Desktop', 'DRIVE')
training_dir = dataset_dir / 'training'
test_dir = dataset_dir / 'test'

training_set = DataSet(training_dir)
test_set = DataSet(test_dir)

def make_visualise(in_path, out_dir, mask_path=None):
    print(f'Processing {in_path.name}...')
    image = im.read(in_path)
    if mask_path is not None:
        mask = im.read(mask_path)
        mask = im.erode(mask, times=3)
        image[mask == 0] = 0
    image = im.to_uint8(image)
    out_name = in_path.stem + '.png'
    out_path = Path(out_dir, out_name)
    im.write(image, out_path)


for dataset in (training_set, test_set):
    for sample in dataset.samples:
        out_dir = str(sample.filtered_dir).replace('filtered', 'filtered_vis')
        make_visualise(sample.frangi_path, out_dir, sample.mask_path)
        make_visualise(sample.sobel_path, out_dir)
        make_visualise(sample.laplacian_path, out_dir)
        make_visualise(sample.lab_path, out_dir)
        make_visualise(sample.hsv_path, out_dir)
