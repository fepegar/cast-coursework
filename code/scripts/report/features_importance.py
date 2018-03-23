#!/usr/bin/env python3

"""
Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

repo_dir = Path(__file__).parents[2]
figures_dir = repo_dir / 'latex' / 'figures'
output_path = figures_dir / 'importances.png'

model_path = Path('/', 'tmp', 'model.pkl')
forest = joblib.load(str(model_path))
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
features = [
    'R', 'G', 'B',
    'L*', 'a*', 'b*',
    'H', 'S', 'V',
    'Sobel',
    'Frangi',
    'Laplacian',
]
N = len(features)

# Print the feature ranking
print("Feature ranking:")

for f in range(N):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.bar(range(N), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(N), np.array(features)[indices], rotation=-30)
plt.xlim([-1, N])
plt.savefig(output_path, dpi=400)
# plt.show()
