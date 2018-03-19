#!/usr/bin/env python3

from time import time
from pathlib import Path

from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier

from cast import DataSet

dataset_dir = Path(Path.home(), 'Desktop', 'DRIVE')
training_dir = dataset_dir / 'training'
test_dir = dataset_dir / 'test'

training_set = DataSet(training_dir)
test_set = DataSet(test_dir)

model_path = Path('/', 'tmp', 'model.pkl')



force = False

if model_path.exists() and not force:
    print('Loading model...')
    clf = joblib.load(str(model_path))
else:
    clf = ExtraTreesClassifier()
    start = time()
    print('Training...')
    clf.fit(training_set.X, training_set.y)
    print('Training time:', time() - start, 'seconds')

    print('\nSaving model...')
    joblib.dump(clf, model_path)

start = time()
print('\nTesting...')
scores = []
for sample in test_set.samples:
    y_sample_predicted = clf.predict_proba(sample.X)
    # scores.append(clf.score(sample.X, sample.y))
    sample.save_prediction(y_sample_predicted[:, 1])
print('Testing time:', time() - start, 'seconds')
# print('Score:', np.array(scores).mean())

print('\nSaving segmentations...')
# test_set.save_predictions(y_training_predicted)
