# Poker GTO Solver — NUS MComp Capstone
# 扑克 GTO 求解器 — NUS 计算机硕士毕业项目

> **Last updated:** 2026-07-02 (Phase 1 — Kuhn Poker CFR)
> **Stack:** Python · numpy · matplotlib
> **GitHub:** https://github.com/YouLi128/poker-gto

---

## Project Goal / 项目目标

**EN:** Build a Game Theory Optimal (GTO) solver for poker, starting from the simplest solvable game (Kuhn Poker) and scaling up through Leduc Hold'em toward a simplified No-Limit Hold'em. The core algorithm is Counterfactual Regret Minimization (CFR). The novel research contribution is a **Safe Exploitation** layer on top of the GTO baseline: detect when an opponent systematically deviates from Nash equilibrium and switch to an exploitative strategy that maximises EV against them — without ever being exploitable ourselves.

**中文:** 构建一个扑克 GTO（博弈论最优）求解器，从最简单的可解游戏（Kuhn Poker）出发，逐步扩展到 Leduc Hold'em 和简化版无限注德州扑克。核心算法是反事实遗憾最小化（CFR）。研究贡献是在 GTO 基线之上加入**安全剥削**层：检测对手何时系统性地偏离纳什均衡，切换到可最大化对弱对手期望值（EV）的策略——同时保证自身永不被剥削。

---

## Roadmap / 阶段计划

| Phase | Content | Status |
|-------|---------|--------|
| **1** | Kuhn Poker — Vanilla CFR · convergence to Nash · verify -1/18 game value | ✅ Done |
| **2** | Leduc Hold'em — larger game tree · card abstraction intro | 🔜 Next |
| **3** | Opponent modelling — detect deviation from Nash · classify opponent type | |
| **4** | Safe Exploitation — adaptive strategy switching · theoretical guarantees | |
| **5** | Simplified NLHE — bet abstraction · heads-up evaluation | |

---

## Progress Log / 进度记录

### 2026-07-02 — Kuhn Poker CFR Complete (Phase 1)

**EN:** Implemented vanilla CFR for Kuhn Poker (3 cards: J < Q < K, 2 players, 1-chip ante). After 50 000 iterations the solver converges to game value −0.05609 vs theoretical Nash equilibrium −0.05556 (error < 0.001). All 12 information sets converge to a valid Nash equilibrium. The CFR implementation follows the Neller-Lanctot convention: `cfr()` returns utility for `current_player(history)`, child utilities are negated on return (switching perspective), and regrets are weighted by the **opponent's** reach probability (counterfactual weighting). Average strategy (not current strategy) converges to Nash. Visualisation: convergence plot + strategy bar chart saved to `kuhn_results.png`.

**中文:** 完成 Kuhn Poker（三张牌 J<Q<K，两名玩家，每人前注 1 筹码）的 vanilla CFR 实现。50,000 次迭代后，求解器收敛到博弈价值 −0.05609，理论纳什均衡为 −0.05556，误差 < 0.001。全部 12 个信息集收敛到有效纳什均衡。CFR 实现遵循 Neller-Lanctot 约定：`cfr()` 返回 `current_player(history)` 的效用，子节点返回值取反（切换视角），遗憾值按**对手**到达概率加权（反事实加权）。平均策略（而非当前策略）收敛到纳什均衡。

**Key result / 核心结果:**

| Info Set | Pass/Check/Fold | Bet/Call | Interpretation |
|----------|----------------|----------|----------------|
| J (P0, root) | 0.779 | 0.221 | Bluff ~22% with worst hand |
| Q (P0, root) | 1.000 | 0.000 | Always check with Q |
| K (P0, root) | 0.339 | 0.661 | Value bet / slow-play mix |
| Jb (P1 faces bet, holds J) | 1.000 | 0.000 | Always fold worst hand |
| Qb (P1 faces bet, holds Q) | 0.662 | 0.338 | Call ~1/3 with Q |
| Kb (P1 faces bet, holds K) | 0.000 | 1.000 | Always call best hand |
| Jp (P1 after P0 check, J) | 0.667 | 0.333 | Probe-bluff 1/3 |
| Qp (P1 after P0 check, Q) | 1.000 | 0.000 | Check back with Q |
| Kp (P1 after P0 check, K) | 0.000 | 1.000 | Value bet K |

---

## What is CFR? / 什么是 CFR？

**EN:** At every decision point, CFR asks: *"If I had chosen a different action, how much more would I have won?"* That difference is **regret**. Regrets accumulate over many iterations; at each step, the strategy assigns more probability to actions with higher accumulated regret (regret matching). The key insight: the **average strategy** across all iterations converges to a Nash equilibrium, even though the current strategy oscillates.

**中文:** 在每个决策点，CFR 问：「如果我当时选了另一个动作，会多赢多少？」这个差值就是**遗憾**。遗憾在多次迭代中累积；每步策略按累积遗憾比例分配动作概率（遗憾匹配）。核心洞察：所有迭代的**平均策略**收敛到纳什均衡，即使当前策略在震荡。

**反事实权重 / Counterfactual weighting:** The regret for player i at a node is weighted by the opponent's reach probability — "how likely was my opponent to have played to reach this node?" This is the "counterfactual" in CFR: we ask what would have happened in a world where player i always played to reach this node.

---

## File Structure / 文件结构

```
poker-gto/
│
├── PROJECT.md
├── requirements.txt
│
├── solver/
│   ├── __init__.py
│   ├── kuhn.py          ← Kuhn Poker rules, payoffs, terminal detection
│   ├── cfr.py           ← KuhnCFR: regret matching, CFR traversal, training loop
│   └── visualize.py     ← convergence plot + strategy bar chart
│
├── run_kuhn.py          ← entry point: train + print + plot
└── kuhn_results.png     ← output
```

---

## Quick Start / 快速开始

```bash
pip install -r requirements.txt

# Default: 10 000 iterations
python run_kuhn.py

# More iterations for tighter convergence
python run_kuhn.py --iterations 100000

# Save plot without opening window
python run_kuhn.py --no-show
```

---

## Known Limitations / 已知局限

**1. Kuhn Poker only**
Full NLHE is computationally intractable without abstraction. Each subsequent phase requires bet-size abstraction (discretising continuous bet sizes into a finite set), which introduces approximation error.

**2. Vanilla CFR convergence rate is O(1/√T)**
CFR+ and Discounted CFR (DCFR) converge faster in practice but are more complex to implement. Phase 2 will upgrade to CFR+.

**3. No exploitability metric yet**
Game value convergence is a proxy for Nash distance. A proper exploitability calculation (compute best response for each player against the other's fixed strategy) will be added in Phase 2.
