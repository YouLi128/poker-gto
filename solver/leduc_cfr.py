"""
CFR solver for Leduc Hold'em.

Same vanilla CFR logic as KuhnCFR, extended for:
  - Two rounds of betting separated by a community card (chance node)
  - 6-card deck with 30 possible (P0, P1) private card deals
  - Info sets include board card in round 2
"""

from collections import defaultdict
from itertools import permutations

import numpy as np

from .leduc import (
    DECK, ACTIONS, NUM_ACTIONS,
    is_terminal, is_chance_node, current_player,
    get_info_set, get_payoff,
)


class LeducCFR:
    def __init__(self):
        self.regret_sum   = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def get_strategy(self, infoset: str) -> np.ndarray:
        pos   = np.maximum(self.regret_sum[infoset], 0.0)
        total = pos.sum()
        return pos / total if total > 0 else np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def get_average_strategy(self, infoset: str) -> np.ndarray:
        total = self.strategy_sum[infoset].sum()
        return self.strategy_sum[infoset] / total if total > 0 else np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def cfr(self, cards: tuple, history: str, p0: float, p1: float) -> float:
        """
        Returns utility for current_player(history).
        cards = (p0_card, p1_card, board_or_None)
        """
        if is_terminal(history):
            p0_util = get_payoff(cards, history)
            return p0_util if current_player(history) == 0 else -p0_util

        # Chance node: deal board card uniformly from remaining 4 cards
        if is_chance_node(history):
            used      = {cards[0], cards[1]}
            remaining = [c for c in DECK if c not in used]
            total = sum(
                self.cfr((cards[0], cards[1], c), history + '/', p0, p1)
                for c in remaining
            )
            return total / len(remaining)

        player  = current_player(history)
        infoset = get_info_set(cards, history, player)
        strategy = self.get_strategy(infoset)

        own_reach = p0 if player == 0 else p1
        self.strategy_sum[infoset] += own_reach * strategy

        action_utils = np.zeros(NUM_ACTIONS)
        for i, action in enumerate(ACTIONS):
            if player == 0:
                action_utils[i] = -self.cfr(cards, history + action,
                                             p0 * strategy[i], p1)
            else:
                action_utils[i] = -self.cfr(cards, history + action,
                                             p0, p1 * strategy[i])

        node_util = strategy @ action_utils
        cf_reach  = p1 if player == 0 else p0
        self.regret_sum[infoset] += cf_reach * (action_utils - node_util)

        return node_util

    def train(self, n_iterations: int = 1000) -> list:
        """
        Iterate CFR over all 30 ordered (P0, P1) card deals.
        Returns cumulative-average game values per iteration.
        """
        all_deals = [(p0, p1) for p0 in DECK for p1 in DECK if p0 != p1]
        running_total = 0.0
        game_values   = []

        for t in range(1, n_iterations + 1):
            iter_val = sum(
                self.cfr((p0c, p1c, None), '', 1.0, 1.0)
                for p0c, p1c in all_deals
            ) / len(all_deals)

            running_total += iter_val
            game_values.append(running_total / t)

            if t % 100 == 0 or t == 1:
                print(f'  iter {t:>5}  game_value = {game_values[-1]:+.5f}')

        return game_values

    def get_all_strategies(self) -> dict:
        return {iset: self.get_average_strategy(iset)
                for iset in sorted(self.strategy_sum)}
