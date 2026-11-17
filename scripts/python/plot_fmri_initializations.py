"""Plot the saved local comparison of 10 versus 100 fMRI initializations."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/reproduction_100seeds/comparison_10_initializations.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/figures/fmri_10_vs_100.png'))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, name in zip(axes.flat, ['RestROIs', 'RestTOIs', 'TaskROIs', 'TaskTOIs']):
        rows = data.loc[data.dataset == name].sort_values('k')
        if rows.k.tolist() != list(range(2, 11)):
            raise ValueError(f'Incomplete K grid for {name}')
        ax.plot(rows.k, rows.SDB_cmeans_10, 'o--', color='#7b8794', label='10 initializations', ms=4)
        ax.plot(rows.k, rows.SDB_cmeans_current, 'D-', color='#176c9d', label='100 initializations', ms=4, markerfacecolor='white')
        ax.set(title=name, xlabel='Number of clusters K', ylabel='Mean SDB (log scale)', yscale='log', xticks=range(2, 11))
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('fMRI: 10 versus 100 initializations per subject and K\nSeed mean per subject, then subject mean; original unscaled data', fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(args.output.with_suffix('.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
