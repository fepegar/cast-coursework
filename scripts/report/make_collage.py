from pathlib import Path

from skimage import io
import matplotlib.pyplot as plt

from cast import test_set
from cast import image as im


repo_dir = Path(__file__).parents[2]
figures_dir = repo_dir / 'latex' / 'figures'
output_path = figures_dir / 'collage.png'

# Previously measured
best = test_set.samples[1]
median = test_set.samples[12]
worst = test_set.samples[2]

samples = best, median, worst


fig, axes = plt.subplots(len(samples), 5, sharex=True, sharey=True)

for axis in fig.axes:
    axis.set_axis_off()

for row, sample in enumerate(samples):
    axes[row, 0].imshow(sample.rgb_data)
    prediction = io.imread(sample.prediction_prob_path).astype(float)
    prediction /= prediction.max()
    axes[row, 1].imshow(prediction)
    driu = io.imread(sample.driu_path).astype(float)
    driu /= driu.max()
    axes[row, 2].imshow(im.compare(sample.labels, prediction > 0.5))
    axes[row, 3].imshow(driu)
    axes[row, 4].imshow(im.compare(sample.labels, driu > 0.5))

# This does not seem to work
plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                    wspace=0.05, hspace=0.05)

fig.savefig(output_path, dpi=400, bbox_inches='tight')
