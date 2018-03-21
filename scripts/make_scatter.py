import matplotlib.pyplot as plt

from cast import test_set

dices = [s.dice_score() for s in test_set.samples]
dices_driu = [s.dice_score(s.get_driu_confusion_map())
              for s in test_set.samples]

fig, axis = plt.subplots()
axis.grid(True)
axis.scatter(dices, dices_driu, 5, zorder=10)
axis.plot((0, 1), (0, 1), alpha=0.25, color='gray')
axis.set_aspect('equal')
axis.set_xlabel('Extra-Trees')
axis.set_ylabel('DRIU')
fig.savefig('/tmp/dices.png', dpi=400)
