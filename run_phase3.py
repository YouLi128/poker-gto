#!/usr/bin/env python3
"""
Phase 3: Opponent Modelling + Safe Exploitation

For each scripted opponent type:
  1. Train Nash (Phase 2) → get GTO strategy
  2. Compute best response to opponent's known strategy (upper bound)
  3. Simulate N hands with GTO P0 vs scripted P1, observe opponent actions
  4. After every K hands, re-estimate opponent strategy and compute
     updated best response → exploit strategy
  5. Plot: GTO EV vs Exploit EV vs Best-Response ceiling
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from solver.leduc_cfr import LeducCFR
from solver.opponent  import OpponentModel, make_scripted_opponent, simulate_game
from solver.exploit   import compute_best_response, evaluate_strategy

STYLES      = ['gto', 'calling_station', 'nit', 'maniac']
STYLE_NAMES = {
    'gto':             'GTO (baseline)',
    'calling_station': 'Calling Station',
    'nit':             'Nit (over-folder)',
    'maniac':          'Maniac (over-bettor)',
}
COLOURS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']


def run_experiment(
    nash_strategy:    dict,
    opp_style:        str,
    n_hands:          int   = 2000,
    update_interval:  int   = 50,
    br_iter:          int   = 200,
    seed:             int   = 42,
) -> dict:
    """
    Simulate n_hands against a scripted opponent.
    Every `update_interval` hands, update opponent model and best response.

    Returns dict with EV arrays for GTO, Exploit, and BR ceiling.
    """
    rng          = np.random.default_rng(seed)
    p1_strategy  = make_scripted_opponent(nash_strategy, opp_style)
    opp_model    = OpponentModel(nash_strategy)

    # Pre-compute oracle best response (knows true P1 strategy)
    br_ceiling_ev = evaluate_strategy(
        compute_best_response(p1_strategy, n_iter=br_iter)[1],
        p1_strategy,
    )
    nash_ev = evaluate_strategy(nash_strategy, p1_strategy)

    # Rolling EV tracking
    gto_evs     = []
    exploit_evs = []

    gto_cumsum     = 0.0
    exploit_cumsum = 0.0
    exploit_strat  = dict(nash_strategy)   # start with Nash

    for hand in range(1, n_hands + 1):
        # --- GTO hand ---
        payoff_gto, log = simulate_game(nash_strategy, p1_strategy, rng)
        gto_cumsum += payoff_gto
        gto_evs.append(gto_cumsum / hand)

        # Update opponent model from observations
        for infoset, action_idx in log:
            opp_model.observe(infoset, action_idx)

        # --- Exploit hand ---
        payoff_exploit, _ = simulate_game(exploit_strat, p1_strategy, rng)
        exploit_cumsum   += payoff_exploit
        exploit_evs.append(exploit_cumsum / hand)

        # Every K hands: recompute best response from current opponent model
        if hand % update_interval == 0:
            est_p1 = opp_model.get_estimated_strategy()
            _, br = compute_best_response(est_p1, n_iter=100)
            # Merge with Nash so unvisited info sets don't fall back to random
            exploit_strat = {**nash_strategy, **br}

    return {
        'gto_evs':        np.array(gto_evs),
        'exploit_evs':    np.array(exploit_evs),
        'nash_ev':        nash_ev,
        'br_ceiling':     br_ceiling_ev,
        'mean_deviation': opp_model.mean_deviation(),
    }


def plot_all(results: dict, save_path='phase3_results.png'):
    n_styles = len(results)
    fig      = plt.figure(figsize=(16, 4 * n_styles))
    gs       = gridspec.GridSpec(n_styles, 1, figure=fig, hspace=0.5)

    for row, (style, data) in enumerate(results.items()):
        ax    = fig.add_subplot(gs[row])
        hands = np.arange(1, len(data['gto_evs']) + 1)

        ax.plot(hands, data['gto_evs'],     lw=1.2, color='#4C72B0',
                label='GTO (Nash)')
        ax.plot(hands, data['exploit_evs'], lw=1.2, color='#DD8452',
                label='Exploit (adaptive)')
        ax.axhline(data['br_ceiling'], color='#55A868', ls='--', lw=1.2,
                   label=f'BR ceiling  {data["br_ceiling"]:+.3f}')
        ax.axhline(data['nash_ev'],    color='#4C72B0', ls=':',  lw=1.0,
                   label=f'GTO vs opp  {data["nash_ev"]:+.3f}')

        ax.set_title(f'Opponent: {STYLE_NAMES[style]}'
                     f'  |  mean deviation from Nash = {data["mean_deviation"]:.3f}',
                     fontsize=11)
        ax.set_xlabel('Hands played')
        ax.set_ylabel('Cumulative avg EV (P0)')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 3 — Opponent Modelling & Safe Exploitation', fontsize=13, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nash-iter',    type=int, default=1000)
    parser.add_argument('--hands',        type=int, default=2000)
    parser.add_argument('--update-every', type=int, default=50)
    parser.add_argument('--no-show',      action='store_true')
    args = parser.parse_args()

    # --- Step 1: Train Nash ---
    print(f'Training Nash strategy ({args.nash_iter} iterations)...')
    solver = LeducCFR()
    solver.train(n_iterations=args.nash_iter)
    nash   = solver.get_all_strategies()
    print(f'Nash ready. Info sets: {len(nash)}')

    # --- Step 2: Run experiment for each opponent type ---
    results = {}
    for style in STYLES:
        print(f'\n[{style}] simulating {args.hands} hands...')
        results[style] = run_experiment(
            nash_strategy   = nash,
            opp_style       = style,
            n_hands         = args.hands,
            update_interval = args.update_every,
        )
        r = results[style]
        print(f'  Deviation from Nash : {r["mean_deviation"]:.3f}')
        print(f'  GTO EV vs opponent  : {r["nash_ev"]:+.4f}')
        print(f'  BR ceiling EV       : {r["br_ceiling"]:+.4f}')
        print(f'  Exploit final EV    : {r["exploit_evs"][-1]:+.4f}')

    # --- Step 3: Plot ---
    fig = plot_all(results, save_path='phase3_results.png')
    print('\nPlot saved → phase3_results.png')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
