"""
Opponent modelling for Leduc Hold'em.

What we can observe about the opponent:
  - Whether they bet or check when first to act
  - Whether they call or fold when facing our bet
  (We cannot see their hole card.)

We aggregate observations into 4 position-level statistics and compare
them to the Nash equilibrium values to detect systematic deviations.
"""

from collections import defaultdict

import numpy as np

from .leduc import ACTIONS, NUM_ACTIONS, DECK, is_terminal, is_chance_node
from .leduc import current_player, get_info_set, get_payoff

# -----------------------------------------------------------------------
# Observable position categories (from OUR perspective watching opponent)
# -----------------------------------------------------------------------
#   R1_OPEN   : opponent is first to act in round 1 (we checked)
#   R1_FACING : opponent faces our bet in round 1
#   R2_OPEN   : opponent is first to act in round 2 (we checked)
#   R2_FACING : opponent faces our bet in round 2

POSITIONS = ['R1_OPEN', 'R1_FACING', 'R2_OPEN', 'R2_FACING']


def _get_position(history: str) -> str | None:
    """
    Return the observable position category for an opponent (P1) action,
    or None if this history is a P0 decision / non-decision node.
    """
    if '/' not in history:
        r1, r2 = history, None
    else:
        i = history.index('/')
        r1, r2 = history[:i], history[i + 1:]

    cur = current_player(history)
    if cur != 1:
        return None   # P0's turn — not an opponent action

    if r2 is None:
        # Round 1
        return 'R1_OPEN' if r1 == 'p' else 'R1_FACING'
    else:
        # Round 2
        return 'R2_OPEN' if len(r2) % 2 == 1 and r2[-1:] in ('', 'p') else 'R2_FACING'


def _r2_position(r2: str) -> str:
    """Helper: given round-2 history at P1's turn, return R2_OPEN or R2_FACING."""
    # P1 acts at r2 length 1 (after P0's first R2 action)
    if r2 == 'p':
        return 'R2_OPEN'   # P0 checked, P1 can probe
    return 'R2_FACING'     # P0 bet, P1 faces a bet


# -----------------------------------------------------------------------
# Opponent model
# -----------------------------------------------------------------------

class OpponentModel:
    """
    Tracks opponent action frequencies at each observable position,
    compares to Nash equilibrium to detect exploitable deviations.
    """

    def __init__(self, nash_strategy: dict):
        """
        nash_strategy : {infoset → np.array([p_pass, p_bet])}
                        from Phase 2 LeducCFR.get_all_strategies()
        """
        self.nash = nash_strategy
        # Observed counts per info set: {infoset → [count_p, count_b]}
        self._counts = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def observe(self, infoset: str, action_idx: int):
        """Record one observed opponent action."""
        self._counts[infoset][action_idx] += 1

    def n_samples(self, infoset: str) -> int:
        return int(self._counts[infoset].sum())

    def estimated_strategy(self, infoset: str, min_samples: int = 15) -> np.ndarray:
        """Empirical frequency if enough data, else fall back to Nash."""
        c = self._counts[infoset]
        total = c.sum()
        if total >= min_samples:
            return c / total
        return self.nash.get(infoset, np.ones(NUM_ACTIONS) / NUM_ACTIONS)

    def infoset_deviation(self, infoset: str) -> float:
        """L1 deviation from Nash at one info set (0 = GTO, 1 = max deviate)."""
        if self.n_samples(infoset) < 5:
            return 0.0
        obs  = self.estimated_strategy(infoset, min_samples=5)
        nash = self.nash.get(infoset, np.ones(NUM_ACTIONS) / NUM_ACTIONS)
        return float(np.abs(obs - nash).sum() / 2)

    def mean_deviation(self) -> float:
        """Average deviation across all observed info sets with ≥5 samples."""
        devs = [self.infoset_deviation(i) for i in self._counts
                if self._counts[i].sum() >= 5]
        return float(np.mean(devs)) if devs else 0.0

    def get_estimated_strategy(self) -> dict:
        """Full estimated strategy dict: observed where possible, Nash elsewhere."""
        strategy = dict(self.nash)
        for iset, c in self._counts.items():
            if c.sum() >= 15:
                strategy[iset] = c / c.sum()
        return strategy


# -----------------------------------------------------------------------
# Scripted opponents
# -----------------------------------------------------------------------

def make_scripted_opponent(nash_strategy: dict, style: str) -> dict:
    """
    Return a P1 strategy dict modified from Nash to simulate a deviant player.

    style options:
      'gto'             — pure Nash (baseline)
      'calling_station' — calls / checks too much (fold prob ×0.25)
      'nit'             — folds too much (bet/call prob ×0.25)
      'maniac'          — bets aggressively (bet prob pushed toward 0.9)
      'passive'         — never probes in R2, checks instead
    """
    strategy = {}
    for iset, probs in nash_strategy.items():
        p, b = probs[0], probs[1]

        if style == 'gto':
            strategy[iset] = probs.copy()

        elif style == 'calling_station':
            new_p = max(0.05, p * 0.25)
            strategy[iset] = np.array([new_p, 1 - new_p])

        elif style == 'nit':
            new_b = max(0.05, b * 0.25)
            strategy[iset] = np.array([1 - new_b, new_b])

        elif style == 'maniac':
            new_b = min(0.95, b + (1 - b) * 0.7)
            strategy[iset] = np.array([1 - new_b, new_b])

        elif style == 'passive':
            # In R2 open positions: never probe-bet; elsewhere play Nash
            if '/' in iset and iset.endswith('/p'):
                strategy[iset] = np.array([1.0, 0.0])
            else:
                strategy[iset] = probs.copy()

        else:
            raise ValueError(f'Unknown style: {style!r}')

    return strategy


# -----------------------------------------------------------------------
# Game simulator
# -----------------------------------------------------------------------

def simulate_game(
    p0_strategy: dict,
    p1_strategy: dict,
    rng: np.random.Generator,
) -> tuple[float, list[tuple[str, int]]]:
    """
    Simulate one hand of Leduc Hold'em.
    Returns (p0_net_payoff, [(p1_infoset, action_idx), ...]).
    """
    deck = list(DECK)
    rng.shuffle(deck)
    p0c, p1c = deck[0], deck[1]
    cards   = (p0c, p1c, None)
    history = ''
    log     = []           # opponent (P1) actions we observe

    while not is_terminal(history):
        if is_chance_node(history):
            used      = {p0c, p1c}
            remaining = [c for c in DECK if c not in used]
            board     = rng.choice(remaining)
            cards     = (p0c, p1c, board)
            history  += '/'
            continue

        player  = current_player(history)
        infoset = get_info_set(cards, history, player)
        strat   = p0_strategy.get(infoset, np.ones(NUM_ACTIONS) / NUM_ACTIONS) \
                  if player == 0 else \
                  p1_strategy.get(infoset, np.ones(NUM_ACTIONS) / NUM_ACTIONS)

        strat = np.clip(strat, 0, 1)
        strat = strat / strat.sum()
        action_idx = rng.choice(NUM_ACTIONS, p=strat)

        if player == 1:
            log.append((infoset, action_idx))

        history += ACTIONS[action_idx]

    return get_payoff(cards, history), log
