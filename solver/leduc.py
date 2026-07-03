"""
Leduc Hold'em — game rules and payoffs.

Deck:  Js Jh Qs Qh Ks Kh  (6 cards)
Antes: 1 chip each (pot starts at 2)
Round 1: bet size 2  |  Round 2: bet size 4
Actions: 'p' = check / fold,  'b' = bet / call
History separator: '/' signals start of round 2 (board card already dealt)

Hand ranking: pair > high card; tie-break by rank (J < Q < K)
  pair = private card rank matches board card rank
"""

RANKS  = ['T', 'J', 'Q', 'K', 'A']
DECK   = [f'{r}{s}' for r in RANKS for s in ('s', 'h')]  # Ts Th Js Jh Qs Qh Ks Kh As Ah
ACTIONS     = ['p', 'b']
NUM_ACTIONS = 2

BET = {1: 2, 2: 4}   # round → bet size

_R1_FOLDS    = {'bp', 'pbp'}
_R1_CONTINUE = {'pp', 'bb', 'pbb'}
_R2_FOLDS    = {'bp', 'pbp'}
_R2_SHOWS    = {'pp', 'bb', 'pbb'}


def _split(history: str):
    """Return (round1_history, round2_history_or_None)."""
    if '/' in history:
        i = history.index('/')
        return history[:i], history[i + 1:]
    return history, None


def is_terminal(history: str) -> bool:
    r1, r2 = _split(history)
    if r1 in _R1_FOLDS:
        return True
    if r1 in _R1_CONTINUE and r2 is not None:
        return r2 in _R2_FOLDS or r2 in _R2_SHOWS
    return False


def is_chance_node(history: str) -> bool:
    """True when round 1 just ended and board card hasn't been dealt yet."""
    r1, r2 = _split(history)
    return r1 in _R1_CONTINUE and r2 is None


def current_player(history: str) -> int:
    r1, r2 = _split(history)
    active = r2 if r2 is not None else r1
    return len(active) % 2


def get_info_set(cards: tuple, history: str, player: int) -> str:
    """
    What a player observes at a decision node.
    Round 1: private card + round1 actions so far
    Round 2: private card + board card + '/' + round1 result + '/' + round2 actions so far
    """
    my_card = cards[player]
    r1, r2 = _split(history)
    if r2 is not None:
        board = cards[2]
        return f'{my_card}{board}/{r1}/{r2}'
    return my_card + r1


def _rank(card: str) -> int:
    return RANKS.index(card[0])


def _has_pair(private: str, board: str) -> bool:
    return private[0] == board[0]


def _showdown_winner(p0: str, p1: str, board: str) -> int:
    """0 = P0 wins, 1 = P1 wins, -1 = tie."""
    v0 = (_has_pair(p0, board), _rank(p0))
    v1 = (_has_pair(p1, board), _rank(p1))
    if v0 > v1: return 0
    if v1 > v0: return 1
    return -1


def get_payoff(cards: tuple, history: str) -> float:
    """
    Net payoff for Player 0 at a terminal node.
    Positive = P0 wins chips, negative = P0 loses chips.
    """
    p0c, p1c = cards[0], cards[1]
    board     = cards[2] if len(cards) > 2 else None
    r1, r2    = _split(history)

    # Chips P0 and P1 each have committed after round 1 (including ante)
    base = 3 if r1 in ('bb', 'pbb') else 1   # ante(1) + optional bet(2)

    if r1 == 'bp':  return 1      # P0 bets R1, P1 folds → P0 wins P1's ante
    if r1 == 'pbp': return -1     # P0 folds R1 → P0 loses ante

    # Round 2
    if r2 == 'bp':  return base   # P0 bets R2, P1 folds → P0 wins P1's base
    if r2 == 'pbp': return -base  # P0 folds R2

    # Showdown
    r2_bet = BET[2] if r2 in ('bb', 'pbb') else 0
    total  = base + r2_bet

    w = _showdown_winner(p0c, p1c, board)
    if w ==  0: return  total
    if w ==  1: return -total
    return 0   # tie: split pot, net 0
