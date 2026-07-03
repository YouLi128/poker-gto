# Poker GTO Solver — NUS MComp Capstone
# 扑克 GTO 求解器 — NUS 计算机硕士毕业项目

> **Last updated:** 2026-07-03 (Phase 4 — Safe Exploitation Framework)
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
| **2** | Leduc Hold'em — two-round game · chance node · 384 info sets | ✅ Done |
| **3** | Opponent Modelling — track deviation from Nash · classify opponent type · best response | ✅ Done |
| **4** | Safe Exploitation — adaptive α · theoretical guarantee · exploitability bound | ✅ Done |
| **5** | Simplified NLHE — bet abstraction · heads-up evaluation | 🔜 Next |

---

## Progress Log / 进度记录

### 2026-07-03 — Safe Exploitation Framework Complete (Phase 4)

**EN:** Formalised the exploitation-safety trade-off as a linear interpolation between Nash and best response: σ_SE(α) = (1−α)·σ\* + α·σ_BR. Proved and verified empirically that Exploitability(σ_SE(α)) ≤ α × Exploitability(σ_BR). Implemented adaptive α selection: α\* = min(ε_budget / exploit_mag, δ·√n / k), which scales exploitation aggressively only when (a) the opponent's deviation δ is large, (b) enough hands n have been observed, and (c) the safety budget ε allows it. The √n term mirrors UCB from bandit theory. Key results across three opponent types: Nit has the largest exploitable gap (GTO +0.535 → BR +1.266) but the highest BR exploitability (0.97), so the safety constraint caps α at ~0.10. Calling Station is already near-optimal under GTO (gap only +0.012). The framework correctly selects aggressive α for low-risk BRs and conservative α for high-risk ones.

**中文:** 将剥削-安全权衡形式化为纳什策略与最优响应的线性插值：σ_SE(α) = (1−α)·σ\* + α·σ_BR。理论证明并实验验证：Exploitability(σ_SE(α)) ≤ α × Exploitability(σ_BR)。自适应 α 选择公式：α\* = min(ε_budget / exploit_mag, δ·√n / k)，仅在对手偏差 δ 显著、观测样本 n 充分且安全预算允许时才激进剥削。√n 项类比 Bandit 理论中的 UCB。关键结论：Nit（过折型）剥削空间最大但 BR 可被剥削量高（0.97），安全约束将 α 限制在 ~0.10；Calling Station 在 GTO 下已接近最优，剥削收益仅 +0.012。

**Key results / 核心结果:**

| Opponent | GTO EV | BR EV | Exploit(BR) | Safe α ceiling |
|----------|--------|-------|-------------|----------------|
| Nit | +0.535 | +1.266 | 0.969 | ~0.10 |
| Calling Station | +1.292 | +1.304 | 0.204 | ~0.49 |
| Maniac | +1.185 | +1.241 | 0.207 | ~0.48 |

---

### 2026-07-03 — Opponent Modelling Complete (Phase 3)

**EN:** Implemented opponent action tracking at each observable position (R1/R2 open, R1/R2 facing a bet). OpponentModel computes the L1 deviation between observed action frequencies and Nash equilibrium strategies. Four scripted opponent types implemented: GTO (baseline), Calling Station (fold prob ×0.25), Nit (bet prob ×0.25), Maniac (bet prob pushed to 0.90). Best response computed via one-sided CFR: P1 strategy is frozen, only P0 updates via regret matching. Nash fallback ensures unvisited info sets don't default to random. Key finding: against Nit, exploit strategy achieves +0.884 EV vs +0.535 GTO (+65%), approaching BR ceiling +1.266. Against GTO opponents, exploit strategy correctly degrades (deviating from Nash is harmful).

**中文:** 在每个可观测位置（R1/R2 主动下注、R1/R2 面对下注）追踪对手动作频率。OpponentModel 计算观测频率与纳什均衡策略之间的 L1 偏差。实现四种对手类型：GTO 基线、Calling Station（弃牌概率 ×0.25）、Nit（下注概率 ×0.25）、Maniac（下注概率推至 0.90）。最优响应通过单侧 CFR 计算：P1 策略冻结，仅 P0 通过遗憾匹配更新。未访问信息集回退到纳什策略。关键发现：对 Nit 对手，剥削策略达到 +0.884 EV，高于 GTO 的 +0.535（提升 65%），接近 BR 上限 +1.266。

---

### 2026-07-02 — Leduc Hold'em CFR Complete (Phase 2)

**EN:** Extended CFR to Leduc Hold'em: 6-card deck (Js Jh Qs Qh Ks Kh), two betting rounds (bet sizes 2 and 4), community card dealt between rounds as a chance node. Hand ranking: pair (private rank = board rank) > high card; tie-break by rank. CFR traversal handles chance nodes by averaging over all 4 remaining board cards uniformly. Training iterates over all 30 ordered (P0, P1) private card pairs. Game converges to EV ~+0.070 with 384 information sets — 32× more than Kuhn, demonstrating the exponential growth of game tree complexity with each added feature.

**中文:** 将 CFR 扩展到 Leduc Hold'em：6 张牌（每种花色两张），两轮下注（下注额 2 和 4 筹码），两轮之间发一张公共牌（机会节点）。牌力排名：对子（私牌点数 = 公共牌点数）> 散牌，同级别按点数大小比较。CFR 遍历在机会节点对 4 张剩余公共牌均匀取平均。训练循环遍历全部 30 种有序 (P0, P1) 私牌组合。博弈收敛到 EV ~+0.070，共发现 384 个信息集——是 Kuhn Poker 的 32 倍，直观展示博弈树随特性增加的指数增长。

---

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
