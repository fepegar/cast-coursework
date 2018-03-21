from pathlib import Path

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


def normalise(image):
    image = image.astype(float)
    image -= image.min()
    image /= image.max()
    return image


fig, axes = plt.subplots(len(samples), 6, sharex=True, sharey=True)

for axis in fig.axes:
    axis.set_axis_off()

for row, sample in enumerate(samples):
    labels1 = sample.labels
    axes[row, 0].imshow(sample.rgb_data)
    prediction = normalise(im.read(sample.prediction_prob_path))
    axes[row, 1].imshow(im.grey2rgb(prediction, mask=sample.mask))
    driu = normalise(im.read(sample.driu_path).astype(float))
    axes[row, 2].imshow(im.compare(labels1, prediction > 0.5))
    axes[row, 3].imshow(im.grey2rgb(driu, mask=sample.mask))
    axes[row, 4].imshow(im.compare(labels1, driu > 0.5))
    labels2 = normalise(im.read(sample.manual_2_path)[..., 0])
    axes[row, 5].imshow(im.compare(labels1, labels2))

# This does not seem to work
plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                    wspace=0.05, hspace=0.05)

fig.savefig(output_path, dpi=400, bbox_inches='tight')
