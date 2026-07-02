"""
Kuhn Poker — game rules and payoffs.

Three-card game (J < Q < K), two players, one chip ante each.
Actions: 'p' = pass (check / fold)
         'b' = bet  (bet  / call)

Terminal histories and their meaning:
  'pp'  → both check → showdown
  'bp'  → P0 bets, P1 folds → P0 wins
  'bb'  → P0 bets, P1 calls → showdown
  'pbp' → P0 checks, P1 bets, P0 folds → P1 wins
  'pbb' → P0 checks, P1 bets, P0 calls → showdown

Payoffs follow the Neller-Lanctot convention:
  get_payoff() returns utility for current_player(history),
  so the CFR recursion can negate when switching players.
"""

CARDS       = ['J', 'Q', 'K']
RANK        = {'J': 1, 'Q': 2, 'K': 3}
ACTIONS     = ['p', 'b']
NUM_ACTIONS = 2

_TERMINAL = {'pp', 'bp', 'bb', 'pbp', 'pbb'}


def is_terminal(history: str) -> bool:
    return history in _TERMINAL


def current_player(history: str) -> int:
    return len(history) % 2


def get_payoff(cards: tuple, history: str) -> float:
    """
    Payoff for current_player(history) at a terminal node.

    'bp'  → player=0, P0 wins 1 chip from P1's ante
    'pbp' → player=1, P1 wins 1 chip from P0's ante
    'pp'  → player=0, showdown, winner takes opponent's ante (±1)
    'bb'  → player=0, showdown, winner takes ante + bet (±2)
    'pbb' → player=1, showdown, winner takes ante + bet (±2)
    """
    p0_higher = RANK[cards[0]] > RANK[cards[1]]

    if history == 'bp':
        return 1.0
    if history == 'pbp':
        return 1.0
    if history == 'pp':
        return 1.0 if p0_higher else -1.0
    if history == 'bb':
        return 2.0 if p0_higher else -2.0
    if history == 'pbb':
        return 2.0 if not p0_higher else -2.0
    raise ValueError(f'get_payoff called on non-terminal history: {history!r}')
