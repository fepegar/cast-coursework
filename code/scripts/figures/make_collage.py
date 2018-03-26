from pathlib import Path

import matplotlib.pyplot as plt

from cast import test_set
from cast import image as im


def normalise(image):
    image = image.astype(float)
    image -= image.min()
    image /= image.max()
    return image


def main():
    repo_dir = Path(__file__).parents[3]
    figures_dir = repo_dir / 'latex' / 'figures'
    output_path = figures_dir / 'collage.png'

    # Previously measured
    best = test_set.samples[1]
    median = test_set.samples[12]
    worst = test_set.samples[2]

    samples = best, median, worst

    fig, axes = plt.subplots(3, 6, figsize=(6, 3))

    for row, sample in enumerate(samples):
        labels1 = sample.labels
        prediction = normalise(im.read(sample.prediction_prob_path))
        driu = normalise(im.read(sample.driu_path).astype(float))
        labels2 = normalise(im.read(sample.manual_2_path)[..., 0])

        images = []
        images.append(sample.rgb_data)
        images.append(im.grey2rgb(prediction, mask=sample.mask))
        images.append(im.compare(labels1, prediction > 0.5))
        images.append(im.grey2rgb(driu, mask=sample.mask))
        images.append(im.compare(labels1, driu > 0.5))
        images.append(im.compare(labels1, labels2))

        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.set_axis_off()
            ax.imshow(image)

    fig.subplots_adjust(left=0, bottom=0,
                        right=1, top=1,
                        wspace=0.05,
                        hspace=0.05)

    fig.savefig(output_path, dpi=400)


if __name__ == '__main__':
    main()
