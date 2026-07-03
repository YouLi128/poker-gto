#!/usr/bin/env python3
"""
Phase 4: Safe Exploitation Framework

Experiments:
  A) α sweep — for each opponent type, plot EV and exploitability vs α
  B) Adaptive α — show adaptive_alpha() selecting the right α in practice
  C) Safety guarantee — verify exploitability ≤ α × Exploitability(BR)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from solver.leduc_cfr   import LeducCFR
from solver.opponent    import make_scripted_opponent, simulate_game, OpponentModel
from solver.exploit     import compute_best_response, evaluate_strategy
from solver.safe_exploit import (
    compute_exploitability, make_safe_exploit, adaptive_alpha,
)

STYLES = ['nit', 'calling_station', 'maniac']
COLOURS = {'nit': '#55A868', 'calling_station': '#DD8452', 'maniac': '#C44E52'}
ALPHA_SWEEP = np.linspace(0, 1, 11)


# ---------------------------------------------------------------------------
# Experiment A: α sweep
# ---------------------------------------------------------------------------

def alpha_sweep(nash, p1_styles, br_strategies, nash_ev, n_eval_iter=150):
    """
    For each α ∈ [0,1] and each opponent type, compute:
      - EV of σ_SE(α) vs that opponent
      - Exploitability of σ_SE(α)
    """
    results = {style: {'ev': [], 'exploit': []} for style in p1_styles}

    # Exploitability of each BR strategy (α=1 baseline)
    br_exploits = {}
    for style in p1_styles:
        br_exploit = compute_exploitability(br_strategies[style], nash_ev,
                                            n_iter=n_eval_iter)
        br_exploits[style] = br_exploit
        print(f'  Exploitability(BR vs {style}) = {br_exploit:.4f}')

    for alpha in ALPHA_SWEEP:
        print(f'  α = {alpha:.1f}', end='  ')
        for style in p1_styles:
            se = make_safe_exploit(nash, br_strategies[style], alpha)
            ev = evaluate_strategy(se, p1_styles[style], n_iter=n_eval_iter)

            # Theoretical bound: α × Exploitability(BR)
            exploit = alpha * br_exploits[style]

            results[style]['ev'].append(ev)
            results[style]['exploit'].append(exploit)
            print(f'{style}:EV={ev:+.3f}', end='  ')
        print()

    return results, br_exploits


# ---------------------------------------------------------------------------
# Experiment B: adaptive α in practice
# ---------------------------------------------------------------------------

def adaptive_experiment(nash, opp_style, p1_strategy, br_strategy,
                        nash_ev, br_exploit,
                        n_hands=3000, update_every=100, seed=7):
    """
    Simulate hands, update opponent model, select α adaptively, track EV.
    """
    rng       = np.random.default_rng(seed)
    opp_model = OpponentModel(nash)

    alphas    = []
    evs_adapt = []
    evs_gto   = []
    cumsum_a  = 0.0
    cumsum_g  = 0.0
    cur_strat = dict(nash)

    for hand in range(1, n_hands + 1):
        # Adaptive strategy hand
        pay_a, log = simulate_game(cur_strat, p1_strategy, rng)
        cumsum_a  += pay_a
        evs_adapt.append(cumsum_a / hand)
        for iset, act in log:
            opp_model.observe(iset, act)

        # GTO hand
        pay_g, _ = simulate_game(nash, p1_strategy, rng)
        cumsum_g += pay_g
        evs_gto.append(cumsum_g / hand)

        # Every update_every hands: recompute α and exploit strategy
        if hand % update_every == 0:
            dev   = opp_model.mean_deviation()
            alpha = adaptive_alpha(
                deviation    = dev,
                n_samples    = hand,
                exploit_mag  = max(br_exploit, 0.01),
                eps_budget   = 0.10,
            )
            alphas.append((hand, alpha, dev))
            cur_strat = make_safe_exploit(nash, br_strategy, alpha)

    return {
        'evs_adapt': np.array(evs_adapt),
        'evs_gto':   np.array(evs_gto),
        'alphas':    alphas,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_alpha_sweep(sweep_results, br_exploits, nash_ev, save_path):
    n = len(sweep_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (style, data) in zip(axes, sweep_results.items()):
        col = COLOURS[style]
        evs      = np.array(data['ev'])
        exploits = np.array(data['exploit'])

        ax2 = ax.twinx()
        ax.plot(ALPHA_SWEEP, evs,      color=col,   lw=2,   label='EV vs opponent')
        ax.axhline(nash_ev, color=col, ls=':',  lw=1.2, label=f'Nash EV {nash_ev:+.3f}')
        ax2.plot(ALPHA_SWEEP, exploits, color='grey', lw=1.5, ls='--',
                 label=f'Exploitability (ε ≤ α×{br_exploits[style]:.3f})')

        ax.set_xlabel('α  (0 = GTO,  1 = pure BR)')
        ax.set_ylabel('EV per hand', color=col)
        ax2.set_ylabel('Exploitability bound', color='grey')
        ax.set_title(f'Opponent: {style}', fontsize=11)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 4A — EV vs Exploitability Trade-off  (σ_SE(α) = (1−α)·Nash + α·BR)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_adaptive(adapt_results, style, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False,
                                    gridspec_kw={'height_ratios': [3, 1]})

    data  = adapt_results
    hands = np.arange(1, len(data['evs_adapt']) + 1)

    ax1.plot(hands, data['evs_adapt'], color=COLOURS[style], lw=1.5,
             label='Adaptive σ_SE(α*)')
    ax1.plot(hands, data['evs_gto'],   color='#4C72B0', lw=1.5, ls='--',
             label='GTO (Nash)')
    ax1.set_ylabel('Cumulative avg EV')
    ax1.set_title(f'Phase 4B — Adaptive α  |  Opponent: {style}', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # α trajectory
    if data['alphas']:
        alpha_hands  = [x[0] for x in data['alphas']]
        alpha_values = [x[1] for x in data['alphas']]
        ax2.step(alpha_hands, alpha_values, where='post',
                 color=COLOURS[style], lw=1.5)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('α* (adaptive)')
        ax2.set_xlabel('Hands played')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nash-iter', type=int, default=800)
    parser.add_argument('--br-iter',   type=int, default=250)
    parser.add_argument('--hands',     type=int, default=3000)
    parser.add_argument('--no-show',   action='store_true')
    args = parser.parse_args()

    # --- Nash ---
    print(f'Training Nash ({args.nash_iter} iters)...')
    solver = LeducCFR()
    solver.train(n_iterations=args.nash_iter)
    nash   = solver.get_all_strategies()
    nash_ev = solver.train(n_iterations=20)[-1]   # quick EV estimate vs itself
    print(f'Nash EV ≈ {nash_ev:+.4f}')

    # --- Scripted opponents + best responses ---
    print('\nComputing best responses...')
    p1_styles    = {s: make_scripted_opponent(nash, s) for s in STYLES}
    br_strategies = {}
    br_evs        = {}
    br_exploits   = {}

    for style in STYLES:
        print(f'  BR vs {style}...')
        br_ev, br = compute_best_response(p1_styles[style], n_iter=args.br_iter)
        br_strategies[style] = {**nash, **br}
        br_evs[style]        = br_ev
        br_exploits[style]   = compute_exploitability(
            br_strategies[style], nash_ev, n_iter=150)
        print(f'    BR EV={br_ev:+.4f}  Exploitability={br_exploits[style]:.4f}')

    # --- Experiment A: α sweep ---
    print('\n[Experiment A] α sweep...')
    sweep_results, _ = alpha_sweep(nash, p1_styles, br_strategies, nash_ev,
                                   n_eval_iter=120)
    fig_a = plot_alpha_sweep(sweep_results, br_exploits, nash_ev,
                             save_path='phase4a_sweep.png')
    print('Saved → phase4a_sweep.png')

    # --- Experiment B: adaptive α (on nit — largest exploitable gap) ---
    print('\n[Experiment B] Adaptive α on nit...')
    adapt = adaptive_experiment(
        nash, 'nit', p1_styles['nit'], br_strategies['nit'],
        nash_ev, br_exploits['nit'],
        n_hands=args.hands,
    )
    fig_b = plot_adaptive(adapt, 'nit', save_path='phase4b_adaptive.png')
    print('Saved → phase4b_adaptive.png')

    # --- Summary ---
    print('\n=== Phase 4 Summary ===')
    print(f'{"Opponent":<18} {"Nash EV":>10} {"BR EV":>10} '
          f'{"Exploit(BR)":>14} {"α=0.5 EV":>12}')
    print('-' * 66)
    for style in STYLES:
        ev_half = sweep_results[style]['ev'][5]   # α=0.5
        print(f'{style:<18} {nash_ev:>10.4f} {br_evs[style]:>10.4f} '
              f'{br_exploits[style]:>14.4f} {ev_half:>12.4f}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
