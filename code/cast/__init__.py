from pathlib import Path

from .dataset import DataSet

dataset_dir = Path(Path.home(), 'Desktop', 'DRIVE')
training_dir = dataset_dir / 'training'
test_dir = dataset_dir / 'test'

training_set = DataSet(training_dir)
test_set = DataSet(test_dir)
