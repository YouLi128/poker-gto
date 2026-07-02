#!/usr/bin/env python3
"""Train Kuhn Poker CFR and display results."""

import argparse
import matplotlib.pyplot as plt

from solver.cfr import KuhnCFR
from solver.visualize import plot_all, NASH_VALUE


def main():
    parser = argparse.ArgumentParser(description='Kuhn Poker CFR Solver')
    parser.add_argument('--iterations', type=int, default=10_000,
                        help='CFR iterations (default: 10 000)')
    parser.add_argument('--no-show', action='store_true',
                        help='Save PNG without opening a window')
    args = parser.parse_args()

    print(f'Training Kuhn Poker CFR — {args.iterations:,} iterations...')
    solver = KuhnCFR()
    game_values = solver.train(n_iterations=args.iterations)
    strategies  = solver.get_all_strategies()

    final_val = game_values[-1]
    print(f'\nGame value  (P0) : {final_val:+.6f}')
    print(f'Nash target      : {NASH_VALUE:+.6f}')
    print(f'Error            : {abs(final_val - NASH_VALUE):.6f}')

    print(f'\n{"Info set":<10} {"Pass/Check/Fold":>18} {"Bet/Call":>12}')
    print('-' * 42)
    for iset, probs in strategies.items():
        print(f'{iset:<10} {probs[0]:>18.4f} {probs[1]:>12.4f}')

    fig = plot_all(game_values, strategies, save_path='kuhn_results.png')
    print('\nPlot saved → kuhn_results.png')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
