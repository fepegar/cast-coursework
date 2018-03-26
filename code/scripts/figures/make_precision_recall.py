"""
Based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve

from cast import test_set

repo_dir = Path(__file__).parents[3]
figures_dir = repo_dir / 'latex' / 'figures'
output_path = figures_dir / 'precision-recall.png'

model_path = Path('/', 'tmp', 'model.pkl')
forest = joblib.load(str(model_path))

y_test = test_set.y
y_score = forest.predict_proba(test_set.X)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_score)

fig, axis = plt.subplots()
axis.step(recall, precision, color='b', alpha=0.2,
          where='post')
axis.fill_between(recall, precision, step='post', alpha=0.2,
                  color='b')

axis.set_xlabel('Recall')
axis.set_ylabel('Precision')
axis.set_aspect('equal')
axis.set_ybound(0.0, 1.05)
axis.set_xbound(0.0, 1.0)
axis.set_axisbelow(True)
axis.grid()

fig.subplots_adjust(left=0, bottom=0.1,
                    right=1, top=0.99,
                    wspace=0, hspace=0)

fig.savefig(output_path, dpi=400)
