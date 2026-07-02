"""
Vanilla Counterfactual Regret Minimization for Kuhn Poker.

Key ideas:
  - Information set (infoset): what the current player can observe
    = their card + the history of actions so far
    e.g. 'Kpb' means "I hold K, P0 checked, P1 bet"
  - Regret matching: strategy[action] ∝ max(0, cumulative_regret[action])
  - Average strategy (not current strategy) converges to Nash equilibrium
  - Counterfactual reach weight: opponent's probability of reaching this node
"""

from collections import defaultdict
from itertools import permutations

import numpy as np

from .kuhn import CARDS, ACTIONS, NUM_ACTIONS, is_terminal, current_player, get_payoff


class KuhnCFR:
    def __init__(self):
        self.regret_sum   = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def get_strategy(self, infoset: str) -> np.ndarray:
        """Current strategy via regret matching."""
        pos = np.maximum(self.regret_sum[infoset], 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def get_average_strategy(self, infoset: str) -> np.ndarray:
        """Average strategy — this is what converges to Nash equilibrium."""
        total = self.strategy_sum[infoset].sum()
        if total > 0:
            return self.strategy_sum[infoset] / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    # ------------------------------------------------------------------
    # CFR traversal
    # ------------------------------------------------------------------

    def cfr(self, cards: tuple, history: str, p0: float, p1: float) -> float:
        """
        Recursive CFR pass. Returns utility for current_player(history).

        p0, p1 : reach probabilities for player 0 and player 1.
        The counterfactual reach weight for player i is the OTHER player's
        reach probability — it measures how much the opponent "intended"
        to reach this node regardless of player i's actions.
        """
        if is_terminal(history):
            return get_payoff(cards, history)

        player  = current_player(history)
        infoset = cards[player] + history
        strategy = self.get_strategy(infoset)

        # Accumulate strategy weighted by this player's reach probability
        own_reach = p0 if player == 0 else p1
        self.strategy_sum[infoset] += own_reach * strategy

        # Recurse over each action
        # Child node belongs to the opponent, so we negate its return value
        action_utils = np.zeros(NUM_ACTIONS)
        for i, action in enumerate(ACTIONS):
            if player == 0:
                action_utils[i] = -self.cfr(cards, history + action,
                                             p0 * strategy[i], p1)
            else:
                action_utils[i] = -self.cfr(cards, history + action,
                                             p0, p1 * strategy[i])

        node_util = strategy @ action_utils

        # Counterfactual regret update — weighted by OPPONENT's reach
        cf_reach = p1 if player == 0 else p0
        self.regret_sum[infoset] += cf_reach * (action_utils - node_util)

        return node_util

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_iterations: int = 10_000) -> list:
        """
        Run CFR for n_iterations over all 6 possible card deals.
        Returns list of cumulative-average game values (for P0) per iteration.
        """
        all_deals = list(permutations(CARDS, 2))   # 6 deals, each prob 1/6
        running_total = 0.0
        game_values   = []

        for t in range(1, n_iterations + 1):
            iteration_value = sum(
                self.cfr(deal, '', 1.0, 1.0) for deal in all_deals
            ) / len(all_deals)

            running_total += iteration_value
            game_values.append(running_total / t)   # cumulative average

        return game_values

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_all_strategies(self) -> dict:
        """Average strategy for every information set, sorted."""
        return {
            iset: self.get_average_strategy(iset)
            for iset in sorted(self.strategy_sum)
        }
