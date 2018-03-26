#!/usr/bin/env python3

from time import time
from pathlib import Path

import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier

from cast import DataSet


dataset_dir = Path(Path.home(), 'Desktop', 'DRIVE')
training_dir = dataset_dir / 'training'
test_dir = dataset_dir / 'test'

training_set = DataSet(training_dir)
test_set = DataSet(test_dir)

model_path = Path('/', 'tmp', 'model.pkl')


force = True

if model_path.exists() and not force:
    print('Loading model...')
    clf = joblib.load(str(model_path))
else:
    clf = ExtraTreesClassifier(class_weight='balanced',
                               random_state=42,  # for reproducibility
                               n_jobs=-1,
                               # n_estimators=5,
                              )

    print('Gathering training data...')
    X, y = training_set.X, training_set.y
    print('X:', X.shape)
    y = y > 0

    weight_fg = training_set.get_samples_class_imbalance().mean()
    weight_bg = 1 - weight_fg
    weight = np.empty_like(y)
    weight[y == 0] = weight_bg
    weight[y != 0] = weight_fg
    start = time()
    print(f'Training with balanced weights...')
    clf.fit(X, y, sample_weight=weight)
    print('Training time:', time() - start, 'seconds')

    print('\nSaving model...')
    joblib.dump(clf, model_path)




print('\nApplying on training set...')
scores = []
dices = []
for sample in training_set.samples:
    print(sample.id)

    X, y = sample.X, sample.y
    y = y > 0

    weight_fg = training_set.get_samples_class_imbalance().mean()
    weight_bg = 1 - weight_fg
    weight = np.empty_like(y)
    weight[y == 0] = weight_bg
    weight[y != 0] = weight_fg
    print(f'Imbalance: {sample.get_background_percentage():.2f}')

    y_sample_predicted = clf.predict_proba(sample.X)
    sample.save_prediction(y_sample_predicted[:, 1])

    score = clf.score(X, y, sample_weight=weight)
    scores.append(score)
    print(f'Weighted score: {score:.2f}')

    dice = sample.dice_score()
    dices.append(dice)
    print(f'Dice score: {dice:.2f}')

    p, r = sample.get_precision_recall()
    print(f'Precision: {p:.2f}')
    print(f'Recall: {r:.2f}')

    print()


train_score = np.array(scores).mean()
print('Mean train score:', train_score)

train_dice_score = np.array(dices).mean()
print('Mean train dice score:', train_dice_score)



print()
print()
print()


start = time()
print('\nTesting...')
scores = []
dices = []
for sample in test_set.samples:
    print(sample.id)
    X, y = sample.X, sample.y
    y = y > 0

    weight_fg = test_set.get_samples_class_imbalance().mean()
    weight_bg = 1 - weight_fg
    weight = np.empty_like(y)
    weight[y == 0] = weight_bg
    weight[y != 0] = weight_fg
    print(f'Imbalance: {sample.get_background_percentage():.2f}')

    y_sample_predicted = clf.predict_proba(sample.X)
    sample.save_prediction(y_sample_predicted[:, 1])

    score = clf.score(X, y, sample_weight=weight)
    scores.append(score)
    print(f'Weighted score: {score:.2f}')

    dice = sample.dice_score()
    dices.append(dice)
    print(f'Dice score: {dice:.2f}')

    p, r = sample.get_precision_recall()
    print(f'Precision: {p:.2f}')
    print(f'Recall: {r:.2f}')

    print()

print('Testing time:', time() - start, 'seconds')
test_score = np.array(scores).mean()
print('Mean test score:', test_score)

test_dice_score = np.array(dices).mean()
print('Mean test dice score:', test_dice_score)


print()
print()

print('Scores difference:', train_score - test_score)
print('Dice scores difference:', train_dice_score - test_dice_score)
