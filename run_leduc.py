#!/usr/bin/env python3
"""Train Leduc Hold'em CFR and display results."""

import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from solver.leduc_cfr import LeducCFR


def plot_results(game_values, strategies, save_path='leduc_results.png'):
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    # --- convergence ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(range(1, len(game_values) + 1), game_values,
             color='steelblue', lw=1.5)
    ax1.axhline(game_values[-1], color='crimson', ls='--', lw=1,
                label=f'final  {game_values[-1]:+.4f}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Game value (P0)')
    ax1.set_title('CFR Convergence — Leduc Hold\'em')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- round 1 strategies only (readable subset) ---
    ax2 = fig.add_subplot(gs[1])
    r1_sets = [k for k in strategies if '/' not in k]
    labels  = sorted(r1_sets)
    x  = np.arange(len(labels))
    w  = 0.35
    pp = [strategies[k][0] for k in labels]
    bp = [strategies[k][1] for k in labels]

    ax2.bar(x - w/2, pp, w, label='Pass/Check/Fold', color='#4C72B0')
    ax2.bar(x + w/2, bp, w, label='Bet/Call',        color='#DD8452')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Probability')
    ax2.set_title('Round 1 Strategy (CFR Average)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def main():
    parser = argparse.ArgumentParser(description='Leduc Hold\'em CFR Solver')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='CFR iterations (default: 1000)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    print(f'Training Leduc Hold\'em CFR — {args.iterations:,} iterations...')
    solver = LeducCFR()
    game_values = solver.train(n_iterations=args.iterations)
    strategies  = solver.get_all_strategies()

    print(f'\nFinal game value (P0): {game_values[-1]:+.6f}')
    print(f'Total info sets found: {len(strategies)}')

    print(f'\n--- Round 1 Strategy ---')
    print(f'{"Info set":<10} {"Pass/Check/Fold":>18} {"Bet/Call":>12}')
    print('-' * 42)
    for iset, probs in sorted(strategies.items()):
        if '/' not in iset:
            print(f'{iset:<10} {probs[0]:>18.4f} {probs[1]:>12.4f}')

    fig = plot_results(game_values, strategies)
    print('\nPlot saved → leduc_results.png')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
