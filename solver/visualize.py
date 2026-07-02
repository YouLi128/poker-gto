"""Visualization helpers for Kuhn Poker CFR results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NASH_VALUE = -1 / 18   # theoretical game value for P0


def plot_convergence(game_values: list, ax=None):
    """Cumulative-average game value vs iterations, overlaid with Nash value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    iters = range(1, len(game_values) + 1)
    ax.plot(iters, game_values, color='steelblue', lw=1.5, label='CFR (cumulative avg)')
    ax.axhline(NASH_VALUE, color='crimson', ls='--', lw=1.5,
               label=f'Nash equilibrium ({NASH_VALUE:.5f})')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Game value  (P0 perspective)')
    ax.set_title('CFR Convergence — Kuhn Poker')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_strategy(strategies: dict, ax=None):
    """
    Grouped bar chart: pass vs bet probability for each information set.
    Info sets are sorted: P0 root actions first (J/Q/K),
    then P1 responses, then P0 responses after check-bet.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Custom sort order to group by street
    order = ['J', 'Q', 'K',          # P0 opening action
             'Jp', 'Qp', 'Kp',       # P1 response after P0 check
             'Jb', 'Qb', 'Kb',       # P1 response after P0 bet
             'Jpb', 'Qpb', 'Kpb']    # P0 response after check-bet
    labels = [i for i in order if i in strategies]

    pass_probs = [strategies[i][0] for i in labels]
    bet_probs  = [strategies[i][1] for i in labels]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, pass_probs, w, label='Pass / Check / Fold', color='#4C72B0')
    ax.bar(x + w/2, bet_probs,  w, label='Bet / Call',          color='#DD8452')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Probability')
    ax.set_title('Nash Equilibrium Strategy (CFR Average)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate with values for readability
    for xi, (p, b) in zip(x, zip(pass_probs, bet_probs)):
        if b > 0.01:
            ax.text(xi + w/2, b + 0.02, f'{b:.2f}', ha='center', fontsize=7)

    return ax


def plot_all(game_values: list, strategies: dict, save_path: str = None):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    plot_convergence(game_values, ax=fig.add_subplot(gs[0]))
    plot_strategy(strategies,    ax=fig.add_subplot(gs[1]))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
