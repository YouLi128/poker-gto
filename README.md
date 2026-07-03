# Poker GTO Solver — Safe Exploitation Framework

**NUS MComp Capstone Project**

A Game Theory Optimal (GTO) poker solver built on Counterfactual Regret Minimization (CFR), extended with a novel **Safe Exploitation** framework that adaptively deviates from Nash equilibrium to exploit opponent weaknesses — with a formal worst-case guarantee.

---

## What this does

| Phase | What it solves |
|-------|---------------|
| **1 — Kuhn Poker CFR** | Finds Nash equilibrium in a 3-card toy game. Verifies convergence to the theoretical value of −1/18. |
| **2 — Leduc Hold'em CFR** | Extends CFR to a two-round game with a community card (chance node). 384 information sets. |
| **3 — Opponent Modelling** | Tracks opponent action frequencies, measures L1 deviation from Nash, computes best response via one-sided CFR. |
| **4 — Safe Exploitation** | Linear mixing σ_SE(α) = (1−α)·Nash + α·BR with adaptive α selection and provable exploitability bound. |

---

## Core idea

Pure GTO is unexploitable but leaves money on the table against weak opponents.  
Pure best response maximises EV against a known opponent but can itself be exploited.

This project finds the middle ground:

```
σ_SE(α) = (1 − α) · σ*  +  α · σ_BR
```

**Guarantee:** `Exploitability(σ_SE(α)) ≤ α × Exploitability(σ_BR)`

**Adaptive α:**
```
α* = min( ε_budget / exploit_mag,  δ · √n / k )
```
- `δ` — opponent's measured deviation from Nash  
- `n` — hands observed (confidence grows with √n, mirroring UCB)  
- `ε_budget` — maximum acceptable EV loss to a worst-case adversary

---

## Key results

**α sweep vs Nit opponent (over-folder):**

| α | EV vs Nit | Exploitability bound |
|---|-----------|---------------------|
| 0.0 (GTO) | +0.535 | 0.000 |
| 0.5 | +0.906 | 0.484 |
| 1.0 (pure BR) | +1.266 | 0.969 |

GTO EV against Nit: **+0.535**  
Exploit EV (adaptive α ≈ 0.10): **+0.610** (+14%, worst-case loss ≤ 0.10)  
BR ceiling: **+1.266**

**Opponent type comparison:**

| Opponent | Deviation from Nash | GTO EV | BR Ceiling | BR Exploitability |
|----------|-------------------|--------|-----------|------------------|
| Nit | 0.41 | +0.535 | +1.266 | 0.969 |
| Calling Station | 0.37 | +1.292 | +1.304 | 0.204 |
| Maniac | 0.36 | +1.185 | +1.241 | 0.207 |

The framework correctly identifies that Calling Station's BR is already close to GTO (gap +0.012) and that Nit's BR, while high-gain, comes with high exploitability — so α is constrained accordingly.

---

## Quick start

```bash
pip install -r requirements.txt

# Phase 1: Kuhn Poker
python run_kuhn.py --iterations 50000

# Phase 2: Leduc Hold'em
python run_leduc.py --iterations 1000

# Phase 3: Opponent modelling
python run_phase3.py --nash-iter 1000 --hands 2000

# Phase 4: Safe exploitation (full experiment)
python run_phase4.py --nash-iter 800 --hands 3000
```

---

## Project structure

```
poker-gto/
├── solver/
│   ├── kuhn.py          Kuhn Poker rules & payoffs
│   ├── cfr.py           Vanilla CFR (KuhnCFR class)
│   ├── leduc.py         Leduc Hold'em rules & payoffs
│   ├── leduc_cfr.py     CFR with chance nodes (LeducCFR class)
│   ├── opponent.py      OpponentModel · scripted opponents · game simulator
│   ├── exploit.py       Best response via one-sided CFR · exact EV evaluation
│   └── safe_exploit.py  σ_SE(α) construction · exploitability · adaptive α
├── run_kuhn.py
├── run_leduc.py
├── run_phase3.py
└── run_phase4.py
```

---

## Why Leduc Hold'em and not real poker?

Full No-Limit Texas Hold'em has a game tree on the order of 10¹⁶⁰ nodes — unsolvable without massive cloud compute and storage. Leduc Hold'em is the standard academic benchmark for poker AI research (used in Libratus, DeepStack papers). The CFR algorithm and Safe Exploitation framework are game-agnostic; Leduc is the proving ground, not the limit.

---

## Research contribution

The Safe Exploitation framework provides a formal answer to the question:  
*"How much should I deviate from GTO given M hands of evidence about my opponent?"*

The adaptive α formula with the √n confidence term and the ε-budget safety constraint is the novel contribution. It unifies bandit-style exploration-exploitation with game-theoretic safety, and generalises to any two-player zero-sum game beyond poker.

---

**Stack:** Python · NumPy · Matplotlib  
**Author:** YouLi128 · NUS MComp Game Theory Capstone
