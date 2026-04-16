# Madison RL Intelligence Agent: Technical Report

## System Architecture

### Overview

The Madison RL Intelligence Agent implements a hierarchical reinforcement learning system
for multi-agent research orchestration. The architecture separates two orthogonal decision
problems into two independent learners:

1. **PPO Meta-Controller** (Strategic Level): Selects *which* specialized agent to activate
   each step. Action space: 4 discrete choices (one per agent role).
2. **Contextual Bandits** (Tactical Level): Given the activated agent, selects *which tool*
   that agent should use. Each agent has its own LinUCB bandit with 6 arms (one per tool).
3. **Transfer Learning** (Adaptation Level): A 4-action PPO policy trained on one query
   domain is fine-tuned on a new domain using a reduced learning rate, transferring
   domain-agnostic research strategies while adapting surface-level preferences.

The two-level split is deliberate: PPO's importance ratio is computed over the 4-agent
action space that its network actually sampled. The bandit optimises the 6-arm tool space
independently. Conflating them into a single 24-action PPO (as earlier implementations
did) produces incorrect importance ratios and degraded learning — a critical architectural
bug that was identified and fixed.

### Data Flow

```
Query
  │
  ▼
PPO Meta-Controller  ──────────────────────────────────────────
  │  Input: 396-dim state (query embedding + coverage + budget)  │
  │  Output: agent_idx ∈ {0,1,2,3}                              │
  │  log_prob, value stored for PPO update                       │
  ▼                                                             │
Agent Role Selected (Search / Evaluator / Synthesis / DeepDive) │
  │                                                             │
  ▼                                                             │
Per-Agent Contextual Bandit (LinUCB + Novelty Bonus)           │
  │  Input: first 128 dims of state as context                  │
  │  Arms:  6 tools per agent                                   │
  │  Score: UCB_i(ctx) + β/√(count(arm,bucket)+1)              │
  │  Output: tool_idx ∈ {0,…,5}                                │
  ▼                                                             │
Composed Environment Action                                     │
  │  action = agent_idx × n_tools + tool_idx                    │
  │  (env accepts 24 discrete actions)                          │
  ▼                                                             │
ResearchEnv.step()                                              │
  │  1. agent_instance.process(memory, tool) called             │
  │     ├─ SearchAgent: coverage-gap analysis → target subtopic │
  │     ├─ EvaluatorAgent: scores unscored sources in memory    │
  │     ├─ SynthesisAgent: identifies synthesisable topics      │
  │     └─ DeepDiveAgent: targets weakest / contradicted topic  │
  │  2. relevance = clip(0.6·affinity + 0.4·confidence + ε)    │
  │  3. Reward computed, next state returned                    │
  ▼                                                             │
Learning Updates  ◄────────────────────────────────────────────┘
  ├─ PPO update when buffer ≥ batch_size (agent_idx stored)
  └─ Bandit update: online ridge regression per tool pull
```

### Component Details

#### PPO Meta-Controller

| Parameter | Value |
|---|---|
| Input dimension | 396 (query 384D + coverage 8D + scalars 4D) |
| **Output actions** | **4 (agent selection only)** |
| Shared backbone | 2 × 256 hidden, ReLU |
| Policy head | Linear(256 → 4) + Softmax |
| Value head | Linear(256 → 1) |
| Learning rate | 3 × 10⁻⁴ |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Gradient norm clip | 0.5 |
| PPO epochs per update | 4 |
| Batch size | 64 |

#### Contextual Bandits (LinUCB + Novelty Bonus)

| Parameter | Value |
|---|---|
| Algorithm | LinUCB with intrinsic motivation |
| Context dimension | 128 (first 128 dims of observation) |
| Arms per bandit | 6 (one per compatible tool set) |
| Exploration α | 1.0 |
| Regularisation λ | 1.0 |
| Novelty scale β | 0.1 |
| Novelty bits | 8 (random-projection LSH) |

One bandit instance per agent role. Bandits are independent: tool knowledge learned
by the search bandit does not transfer to the evaluator bandit.

#### Transfer Learning

Pre-train a 4-action PPO on a source query domain, then fine-tune on the target domain:

- **Network copy**: `target_ppo.load_state_dict(deepcopy(source_ppo.state_dict()))`
- **Reduced LR**: all parameter groups set to `0.3 × 3×10⁻⁴ = 9×10⁻⁵`
- **No layer freezing**: all parameters remain trainable; the lower LR prevents
  catastrophic forgetting of source-domain agent-coordination priors
- **Fresh bandits**: target-domain bandits re-initialise, re-learning tool preferences
  from scratch (tool preferences are domain-specific; agent-coordination is not)

---

## Mathematical Formulation

### PPO (Proximal Policy Optimization)

#### Policy Gradient Objective

```
L^CLIP(θ) = Ê[ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
```

Where:
- `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)` — probability ratio over **agent index** (4 classes)
- `Â_t` — advantage estimate via GAE
- `ε = 0.2` — clipping parameter

The action `a_t` stored in the rollout buffer is `agent_idx ∈ {0,1,2,3}`, matching
what the network sampled. The tool index chosen by the bandit is never stored in the
PPO buffer.

#### Generalized Advantage Estimation (GAE)

```
Â_t^GAE(γ,λ) = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

With γ = 0.99 and λ = 0.95.

#### Value Function Loss

```
L^VF(θ) = Ê[ (V_θ(s_t) − V_t^target)² ]
V_t^target = Â_t + V_θ_old(s_t)
```

#### Full Objective

```
L(θ) = L^CLIP(θ) − c_1 L^VF(θ) + c_2 H[π_θ](s_t)
```

Where `c_1 = 0.5` (value loss coefficient) and `c_2 = 0.01` (entropy bonus).

---

### LinUCB Contextual Bandit

#### Upper Confidence Bound

For arm `a` and context `x_t ∈ ℝ^{128}`:

```
UCB_a(x_t) = θ_a^T x_t  +  α √(x_t^T A_a^{−1} x_t)
              ─────────────   ─────────────────────────
              predicted        exploration bonus
              reward           (large when uncertain)
```

Where:
- `A_a = λI + Σ_τ x_τ x_τ^T` — regularised design matrix
- `b_a = Σ_τ r_τ x_τ` — accumulated reward-context products
- `θ_a = A_a^{−1} b_a` — ridge regression estimate
- `α = 1.0` — exploration parameter

#### Incremental Update (Sherman-Morrison)

After pulling arm `a` with context `x` and observing reward `r`:

```
A_a ← A_a + x x^T
b_a ← b_a + r x
A_a^{−1} ← A_a^{−1} − (A_a^{−1} x x^T A_a^{−1}) / (1 + x^T A_a^{−1} x)
```

The Sherman-Morrison formula updates `A^{−1}` in O(d²) instead of O(d³),
making online updates practical for context_dim = 128.

---

### Intrinsic Motivation: Novelty Bonus

Layered on top of LinUCB to encourage exploration of under-visited
(arm, context) regions:

#### Random-Projection Hashing

A fixed projection matrix `P ∈ ℝ^{n_bits × d}` (sampled once, never updated):

```
bits(x) = sign(P x)  ∈ {0,1}^{n_bits}
bucket(arm, x) = (arm,  pack_bits(bits(x)))
```

With n_bits = 8, there are 256 possible context buckets per arm (2048 total).
The Johnson-Lindenstrauss lemma guarantees that similar contexts hash to the
same bucket with high probability.

#### Novelty Bonus

```
novelty(arm, x) = β / √(count(bucket(arm, x)) + 1)
```

Where:
- `β = 0.1` — bonus scale
- `count(·)` — visit count for that bucket, incremented after each selection

#### Combined Selection Score

```
score_i(x) = UCB_i(x) + novelty(i, x)
```

The bonus decays as `1/√N`, so it approaches zero as experience accumulates —
exploration automatically transitions to exploitation without tuning a decay schedule.

---

### Transfer Learning Formulation

#### Fine-Tuning Objective

Standard RL objective on target domain with pre-trained initialisation:

```
L_transfer(θ) = L_PPO(θ; D_target)
```

Starting from `θ_0 = θ_source` (deep copy). No distillation term — the lower
learning rate (0.3×) is sufficient to preserve pre-trained priors during early
fine-tuning steps.

#### Why Agent-Selection Transfers

The 4-action PPO learns *when* to search vs synthesise vs evaluate — strategies
that are domain-agnostic (budget management, coverage gap prioritisation). Only
*which tool works best* is domain-specific, and that knowledge lives in the bandits
(which re-initialise for the target domain). This clean separation is what makes
transfer positive rather than neutral or negative.

---

## Reward Function

### Per-Step Reward

```
R_step = w_r · relevance
       + w_c · coverage
       − w_l · clip(latency / 2, 0, 1)
       + w_d · diversity
       + w_q · (relevance × coverage)
```

| Component | Weight | Description |
|---|---|---|
| Relevance | 1.0 | Quality of newly retrieved information |
| Coverage | 0.5 | Fraction of subtopics addressed |
| Latency penalty | 0.3 | Cost of slow tool calls |
| Diversity | 0.4 | Spread of source domains |
| Quality | 0.6 | relevance × coverage joint signal |

### Relevance Computation (with Real Agent Execution)

```
relevance = clip( 0.6 × affinity(tool, query_type)
                + 0.4 × agent_confidence(memory_state)
                + ε,   0, 1 )
```

Where `ε ~ N(0, 0.1)` and `agent_confidence` is drawn from the agent's analysis:
- **SearchAgent**: 0.7 (fixed; does not depend on memory state)
- **EvaluatorAgent**: `mean(credibility_scores)` of all sources in memory
- **SynthesisAgent**: `min(1.0, n_synthesisable_topics × 0.2)` — zero early in episodes
- **DeepDiveAgent**: 0.6 (fixed)

### End-of-Episode Bonus

```
R_episode = 2.0 × coverage  (if coverage > 0.7)
          + clip(n_findings / max(budget_used, 0.1), 0, 2)
          − 1.0  (if diversity < 0.2)
```

---

## Agent Execution in the Training Loop

A key architectural feature is that the four specialized agents run their *actual logic*
during every training step — not just in the interactive demo.

### What Each Agent Does per Step

**SearchAgent** (`_execute`):
- Scans `memory.messages` for EvaluatorAgent credibility warnings
- Accumulates a `avoided_domains` blocklist from those warnings
- Runs `_find_coverage_gaps(memory)` → subtopics with coverage < 0.7
- Returns `target_subtopic` (lowest-coverage gap) used to assign the Finding

**EvaluatorAgent** (`_execute`):
- Collects `unscored = [s for s in memory.sources if s.credibility_score == 0.0]`
- Retrieved sources start with `credibility_score = 0.0`; evaluator fills them in
- Scores up to 5 sources per step using domain heuristics:
  ```
  score(source) = DOMAIN_SCORES[source.domain] × (0.7 + 0.3 × recency_score)
  ```
- Broadcasts `credibility_warning` messages for sources scoring < 0.5

**SynthesisAgent** (`_execute`):
- Filters sources by `credibility_score ≥ 0.6` (after evaluator has run)
- Groups findings by subtopic; synthesisable = subtopics with ≥ 2 findings
- Detects contradictions: subtopics where `std(confidences) > 0.3`
- Returns `contradictions_found` as a list — consumed by DeepDiveAgent

**DeepDiveAgent** (`_execute`):
- Reads SynthesisAgent messages for contradiction topic lists
- Prioritises resolving contradictions over general gap-filling
- Falls back to `min(coverage_map.items())` if no contradictions

### Inter-Agent Messaging

All agents share a `SharedMemory` instance. After each `_execute`, `process()` calls
`_broadcast()` which appends an `AgentMessage` to `memory.messages`. The next agent
activated reads those messages via `_read_messages()` before acting. This creates a
real coordination loop: EvaluatorAgent's credibility warnings influence SearchAgent's
domain avoidance; SynthesisAgent's contradiction list steers DeepDiveAgent's targets.

---

## Design Choices and Implementation

### State Space Design

The 396-dimensional observation vector:

| Slice | Dim | Content |
|---|---|---|
| Query embedding | 384 | Random Gaussian (placeholder for sentence-BERT) |
| Coverage per subtopic | 8 | Per-subtopic coverage score, padded to 8 |
| Budget remaining | 1 | `1 − budget_used / max_budget` |
| Step fraction | 1 | `step_count / max_steps` |
| Sources (normalised) | 1 | `min(n_sources / 20, 1)` |
| Diversity | 1 | `n_unique_domains / n_sources` |

### Action Space Design

The environment accepts 24 discrete actions (`agent_idx × 6 + tool_idx`). PPO only
produces 4-class logits (agent selection). The bandit produces the tool index
independently. The composed action is assembled before calling `env.step()`, but the
two optimisers each only ever see their own sub-space.

### Agent-Tool Compatibility

| Agent | Compatible Tools |
|---|---|
| SearchAgent | web_scraper, api_client, academic_search |
| EvaluatorAgent | credibility_scorer, web_scraper |
| SynthesisAgent | relevance_extractor, pdf_parser |
| DeepDiveAgent | web_scraper, api_client, relevance_extractor |

Incompatible pairings produce `relevance = 0.05` and `latency × 1.5`. The agent still
runs its logic and broadcasts messages (coordination is not penalised), but retrieval
quality is degraded.

### RelevanceExtractor Custom Tool

`src/tools/relevance_extractor.py` implements a TF-IDF + cosine-similarity passage
extractor used by SynthesisAgent and DeepDiveAgent:

1. Segment document into paragraphs / sentence chunks
2. Fit TF-IDF vectoriser (max 5000 features, unigrams + bigrams, sublinear TF)
3. Rank passages by cosine similarity to query vector
4. Return top-k with relevance scores

---

## Experimental Results and Analysis

### Baseline Comparison (Default Environment)

All results: 500 episodes, 3 seeds, mean reward over last 50 episodes.

| Configuration | Mean ± Std | Improvement vs Random | 95% CI |
|---|---|---|---|
| Random Baseline | 7.02 ± 0.21 | — | [6.38, 7.67] |
| Heuristic | 10.74 ± 0.06 | +53.0% | [10.55, 10.94] |
| Bandit Only | 23.32 ± 0.59 | +232.1% | [21.53, 25.12] |
| PPO Only | 27.29 ± 0.43 | +288.6% | [25.98, 28.60] |
| Full System (PPO+Bandit) | **27.48 ± 0.98** | **+291.3%** | [24.50, 30.46] |

Full System achieves the highest mean reward and is the only configuration that
combines strategic agent selection (PPO) with tactical tool optimisation (bandits).

### Statistical Validation

Welch's t-tests comparing each configuration against the Random Baseline
(using per-seed final-50-episode means as the sample, n=3 per condition):

| Comparison | t-statistic | p-value | Significant |
|---|---|---|---|
| Heuristic vs Random | t = 23.73 | p < 0.001 | Yes |
| Bandit Only vs Random | t = 36.74 | p < 0.001 | Yes |
| PPO Only vs Random | t = 59.65 | p < 0.001 | Yes |
| Full System vs Random | t = 28.83 | p < 0.001 | Yes |
| Full System vs PPO Only | t = 0.25 | p = 0.816 | No |

The last row is noteworthy: Full System and PPO Only are statistically indistinguishable
at the final-50-episode window. Both significantly outperform all other baselines. This
reflects that PPO's strategic agent selection is the primary performance driver; the
bandit's contribution is tool-level efficiency rather than order-of-magnitude reward lift.

### Varied Environment Results

#### Tight Budget (`latency_mean=1.0`, `latency_std=0.3`, `max_steps=12`)

Each step costs 2× the default latency; episodes are 40% shorter.

| Configuration | Mean ± Std |
|---|---|
| Random Baseline | 2.60 ± 0.08 |
| Heuristic | 4.50 ± 0.03 |
| Bandit Only | 10.45 ± 0.11 |
| PPO Only | 12.78 ± 0.69 |
| **Full System** | **13.26 ± 0.17** |

Full System wins (+3.8% vs PPO Only) and has notably lower variance (±0.17 vs ±0.69).
Under budget pressure, the bandit's tool specialisation reduces wasted steps —
choosing the right tool matters more when fewer steps are available.

#### Complex Queries (`relevance_noise_std=0.25`, `n_source_pool=20`)

Higher retrieval noise and a smaller source pool make each step less predictable.

| Configuration | Mean ± Std |
|---|---|
| Random Baseline | 7.40 ± 0.34 |
| Heuristic | 10.65 ± 0.25 |
| Bandit Only | 22.89 ± 0.70 |
| **PPO Only** | **26.81 ± 0.35** |
| Full System | 25.08 ± 1.83 |

PPO Only edges Full System here. The higher Full System variance (±1.83) reflects that
with real agent execution, `agent_confidence` fluctuates more under noisy retrieval —
`SynthesisAgent` confidence is near zero early in episodes when little has been gathered,
suppressing relevance. PPO Only's fixed affinity-driven relevance is more stable in this
regime. This is an honest result: the real agent confidence signal has higher variance
than the pure affinity baseline.

### Transfer Learning Results

Pre-train on technical queries (300 episodes) → fine-tune on opinion queries (100 episodes).
Results are mean reward over last 20 episodes on the target domain, 2 seeds.

| Condition | Mean ± Std | Early Ep 1–10 |
|---|---|---|
| No fine-tuning (source model on target) | 26.51 ± 0.19 | 26.65 |
| From scratch (100 ep on target) | 28.73 ± 1.43 | 26.55 |
| **Transfer (pre-train → fine-tune)** | **30.44 ± 0.24** | **31.21** |

The transfer advantage is most visible in early adaptation (+17.6% in episodes 1–10).
The pre-trained policy immediately applies learned agent-coordination strategies to the
new domain; from-scratch training needs those early episodes to discover basic policies.

Final-episode performance: Transfer (30.44) > Scratch (28.73) (+5.9%). With only 2 seeds
the t-test is underpowered (t=1.18, p=0.36), but the consistency across seeds (transfer
std 0.24 vs scratch std 1.43) reflects that the transferred policy is not only better
but more reliable.

### Few-Shot Evaluation Results

K-shot fine-tuning (K ∈ {1, 5, 10, 20}) on the target domain, evaluated over 30 episodes.
Source domain: technical (300 ep pre-train). Target domain: opinion. 3 seeds.

| K | Transfer Mean ± Std | Scratch Mean ± Std | Advantage |
|---|---|---|---|
| 1 | 25.49 ± 1.24 | 22.69 ± 0.16 | **+12.3%** |
| 5 | 25.42 ± 1.27 | 22.96 ± 0.26 | **+10.7%** |
| 10 | 25.44 ± 1.15 | 23.27 ± 0.15 | **+9.3%** |
| 20 | 25.26 ± 1.19 | 23.14 ± 0.33 | **+9.2%** |

The largest advantage is at K=1 (+12.3%), where from-scratch training has had almost no
time to learn anything. Even with a single fine-tuning episode, the pre-trained policy
already outperforms 1-episode scratch training by over 12%. The advantage narrows
slightly as K grows because scratch training accumulates useful signal — but even at
K=20, transfer maintains a +9.2% edge.

This demonstrates that the agent-coordination knowledge encoded in the 4-action PPO
(when to search, synthesise, evaluate, or deep-dive) genuinely transfers across query
types, regardless of surface-level tool preferences.

---

## What the Bandits Learned

From training statistics (100-episode full training run):

### SearchAgent Bandit

| Tool | Pulls | Avg Reward |
|---|---|---|
| web_scraper | 167 (34%) | 1.395 |
| api_client | 155 (32%) | 1.422 |
| academic_search | 168 (34%) | 1.387 |

The search bandit converges to a near-uniform distribution across all three tools,
reflecting that for general research queries the tools are approximately equally
effective. The bandit correctly discovers this and stops over-committing to any one tool.

### EvaluatorAgent Bandit

| Tool | Pulls | Avg Reward |
|---|---|---|
| credibility_scorer | 129 (24%) | 0.722 |
| web_scraper | 408 (76%) | 1.400 |

The evaluator bandit develops a strong preference for `web_scraper` (76% of pulls).
The average reward gap (1.40 vs 0.72) reflects that web_scraper retrieves new sources
with non-zero relevance, while credibility_scorer only re-scores existing sources without
adding new findings. The bandit correctly learns that web_scraper provides a richer
signal in the current reward structure.

### SynthesisAgent Bandit

| Tool | Pulls | Avg Reward |
|---|---|---|
| relevance_extractor | 246 (49%) | 0.831 |
| pdf_parser | 253 (51%) | 0.901 |

Near-uniform split with a slight pdf_parser preference. Both tools produce similar
rewards for synthesis tasks — the bandit settles into a moderate pdf_parser preference
but maintains meaningful exploration of relevance_extractor.

### DeepDiveAgent Bandit

| Tool | Pulls | Avg Reward |
|---|---|---|
| web_scraper | 168 (38%) | 1.463 |
| api_client | 171 (39%) | 1.508 |
| relevance_extractor | 98 (22%) | 0.741 |

The deep-dive bandit clearly avoids `relevance_extractor` (22% pulls, 0.74 avg reward
vs 1.46–1.51 for retrieval tools). DeepDiveAgent's role is to follow up on coverage
gaps by retrieving new sources; relevance_extractor re-processes existing content,
producing lower marginal reward. The bandit correctly identifies this pattern.

---

## Challenges and Solutions

### Challenge 1: PPO-Bandit Action Space Conflict

**Problem**: An initial implementation gave PPO a 24-action space (agent × tool jointly),
then had the bandit independently override the tool dimension. This produced incorrect
importance ratios: `old_log_prob` was computed over PPO's sampled action (index 0–23),
but `new_log_prob` during update used the same index — except the bandit had replaced
the tool portion, so the two referred to different actions. The importance ratio
`exp(new − old)` was meaningless.

**Solution**: Separate action spaces entirely. PPO operates on 4 actions (agent only),
stores `agent_idx` in the rollout buffer. The bandit operates on 6 arms (tool only)
independently. The environment action `agent_idx × n_tools + tool_idx` is assembled
after both decisions, but neither learner's update sees the composed index.

### Challenge 2: Transfer Learning Direction

**Problem**: With a 24-action PPO, the source-domain policy encodes TECHNICAL-specific
(agent, tool) joint patterns. Fine-tuning on OPINION tried to overwrite with OPINION
patterns — negative transfer, from-scratch outperforming transfer.

**Solution**: With 4-action PPO the source policy encodes domain-agnostic agent
coordination. Only the bandits need to re-learn tool preferences for the target domain.
Transfer becomes consistently positive.

### Challenge 3: Transfer Stability (Learning Rate)

**Problem**: Fine-tuning with the default learning rate (3×10⁻⁴) overwrote pre-trained
agent-coordination priors within the first few episodes.

**Solution**: Fine-tuning uses 0.3× LR (9×10⁻⁵) applied to all parameters simultaneously.
No layer freezing is needed — the reduced step size is sufficient to preserve priors
while still allowing adaptation.

### Challenge 4: Credit Assignment

**Problem**: PPO had difficulty attributing rewards to specific agent selections when
episodes have 10–20 steps and rewards are delayed (coverage bonus accrues over time).

**Solution**: GAE with λ=0.95 smoothly interpolates between Monte Carlo returns (λ=1,
unbiased but high variance) and TD(0) (λ=0, low variance but biased). At λ=0.95 the
effective horizon is ~20 steps, matching the episode length.

### Challenge 5: Bandit Exploration Under Sparse Context

**Problem**: LinUCB with a 128-dimensional context was slow to accumulate meaningful
signal in early episodes, leading to near-random tool selection for too long.

**Solution**: The novelty bonus `β/√(count+1)` provides optimism-in-the-face-of-uncertainty
for genuinely unexplored (arm, context) buckets. The 8-bit random projection collapses
the 128-dimensional context to 256 possible buckets per arm, making counts meaningful
much earlier.

---

## Algorithm Complexity Analysis

### PPO Complexity

| Resource | Complexity |
|---|---|
| Parameters | O(obs_dim × hidden + hidden² + hidden × 4) |
| Forward pass | O(obs_dim × hidden + hidden²) |
| GAE computation | O(T) where T = rollout length |
| Update step | O(B × hidden²) per epoch, B = batch size |
| Memory | ~1.1 MB for hidden_dim=256, n_actions=4 |

### Bandit Complexity (per agent)

| Resource | Complexity |
|---|---|
| Storage | O(n_arms × d²) for A matrices, O(n_arms × d) for b vectors |
| UCB computation | O(n_arms × d²) using cached A_inv |
| Update (Sherman-Morrison) | O(d²) vs O(d³) for full inverse |
| Novelty tracker | O(n_arms × 2^n_bits) worst-case bucket space |

For d=128, n_arms=6: storage ≈ 6 × 128² × 8 bytes ≈ 786 KB per bandit.
Sherman-Morrison saves O(d) = 128× over full matrix inversion.

---

## Hyperparameter Sensitivity

### PPO: Learning Rate

| LR | Observed Behaviour |
|---|---|
| 1×10⁻³ | Unstable training, high variance across seeds |
| **3×10⁻⁴** | **Stable convergence, optimal final performance** |
| 1×10⁻⁴ | Slower convergence, similar final performance |

### PPO: Clipping Parameter ε

| ε | Observed Behaviour |
|---|---|
| 0.1 | Conservative updates, slow early learning |
| **0.2** | **Balanced stability and progress** |
| 0.3 | Occasional instability, risk of large policy jumps |

### Bandit: Exploration α

| α | Observed Behaviour |
|---|---|
| 0.5 | Commits too early; misses better tools in rare query types |
| **1.0** | **Balanced exploration-exploitation across all agents** |
| 2.0 | Over-explores; marginal improvement in final tool selection |

### Bandit: Novelty β

| β | Observed Behaviour |
|---|---|
| 0 | Pure LinUCB; slower tool discrimination early in training |
| **0.1** | **Efficient early exploration; decays naturally** |
| 0.5 | Over-explores; novelty dominates UCB signal for ~100 steps |

---

## Implementation Details

### PPO Network

```python
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = 396, n_actions: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)  # 4 agent choices
        self.value_head  = nn.Linear(hidden_dim, 1)
```

### LinUCB Arm (Sherman-Morrison Update)

```python
class LinUCBArm:
    def __init__(self, context_dim: int, regularization: float = 1.0):
        self.A     = np.eye(context_dim) * regularization
        self.b     = np.zeros(context_dim)
        self.A_inv = np.eye(context_dim) / regularization  # cached inverse

    def get_ucb(self, context: np.ndarray, alpha: float) -> float:
        theta = self.A_inv @ self.b
        return theta @ context + alpha * np.sqrt(context @ self.A_inv @ context)

    def update(self, context: np.ndarray, reward: float) -> None:
        self.A += np.outer(context, context)
        self.b += reward * context
        x = context.reshape(-1, 1)
        self.A_inv -= (self.A_inv @ x @ x.T @ self.A_inv) / (
            1.0 + (x.T @ self.A_inv @ x).item()
        )
```

### Novelty Tracker

```python
class NoveltyTracker:
    def __init__(self, context_dim, n_bits=8, beta=0.1, seed=0):
        self.beta  = beta
        self._proj = np.random.default_rng(seed).standard_normal((n_bits, context_dim))
        self._counts: Dict[Tuple[int, int], int] = defaultdict(int)

    def _bucket(self, arm, context):
        bits = (self._proj @ context) >= 0
        return (arm, int(np.packbits(bits, bitorder="big")[0]))

    def bonus(self, arm, context) -> float:
        return self.beta / np.sqrt(self._counts[self._bucket(arm, context)] + 1)

    def update(self, arm, context) -> None:
        self._counts[self._bucket(arm, context)] += 1
```

### Two-Level Decision in Training Loop

```python
# PPO selects agent (4-class)
agent_idx, log_prob, value = ppo.select_action(obs)
agent_name = env.AGENTS[agent_idx].value

# Bandit selects tool for that agent (6-arm)
context = obs[:bandit_config.context_dim]
tool_arm, tool_name = bandits.select_tool(agent_name, context)

# Compose environment action
tool_idx    = tool_names_list.index(tool_name)
final_action = agent_idx * env.n_tools + tool_idx
next_obs, reward, term, trunc, _ = env.step(final_action)

# Store AGENT_IDX in PPO buffer (not final_action)
ppo.store_transition(obs, agent_idx, reward, done, log_prob, value)
bandits.update(agent_name, tool_arm, context, reward)
```

---

## Future Improvements

### Technical Enhancements

1. **Real Query Embeddings**: Replace random Gaussian embeddings with sentence-BERT
   (all-MiniLM-L6-v2) for semantically meaningful 384-dim representations
2. **Multi-Agent PPO (MAPPO)**: Shared critic across all four agent roles for
   coordinated policy gradients
3. **Neural Bandits**: Replace LinUCB with a neural upper-confidence-bound approach
   for non-linear tool-context relationships
4. **MAML for Few-Shot**: Replace fine-tuning with Model-Agnostic Meta-Learning
   for guaranteed few-shot convergence bounds
5. **Attention over Memory**: Replace the fixed-size coverage vector with an
   attention mechanism over `SharedMemory.findings` for richer state representations

### Research Directions

1. **Real API Integration**: Wire `src/tools/real_apis.py` (Wikipedia, Semantic Scholar,
   DuckDuckGo) into the training loop for non-simulated tool feedback
2. **Human-in-the-Loop**: Incorporate human relevance judgements as reward signal
3. **Cross-Agent Transfer**: Transfer tool-selection bandits across agent roles
   (e.g. SearchAgent bandit → DeepDiveAgent bandit warm-start)
4. **Adversarial Robustness**: Test under noisy or misleading source content

---

## Ethical Considerations

### Bias and Fairness

- **Domain Bias**: The tool-query affinity matrix encodes fixed assumptions about which
  tools suit which query types. These could systematically disadvantage certain query
  domains.
- **Source Bias**: Credibility heuristics use domain-based scores (e.g. arxiv.org = 0.9,
  medium.com = 0.4) — a simplification that may not generalise.
- **Mitigation**: Regular audits of bandit pull distributions; fairness constraints in
  the reward function to penalise over-reliance on a single domain.

### Transparency and Explainability

- **PPO Opacity**: Neural policy networks are difficult to interpret directly. Bandit
  pull statistics (`get_all_stats()`) provide partial transparency into tool preferences.
- **Logging**: All agent decisions, tool selections, and reward components are logged
  per-step in `step_info` for post-hoc analysis.

### Responsible AI Practices

- **Safety Constraints**: Budget limits and max-step caps prevent runaway resource
  consumption.
- **Human Oversight**: All generated research summaries require human validation before
  use.
- **Open Source**: Full source code, configuration, and experiment results at
  https://github.com/UshakeShravya/madison-rl-intel

---

## Conclusion

This report documents the Madison RL Intelligence Agent, a hierarchical RL system for
automated research orchestration. The central architectural insight is the **clean
separation of agent selection (PPO, 4 actions) from tool selection (LinUCB bandits,
6 arms per agent)** — a design that keeps importance ratios correct, enables independent
optimisation of strategic and tactical decisions, and makes transfer learning positive
by isolating domain-agnostic coordination knowledge in the PPO.

### Key Empirical Results

| Result | Value |
|---|---|
| Full System vs Random | +291.3% (27.48 vs 7.02 mean reward) |
| Full System vs Heuristic | +156.0% |
| Transfer advantage at K=1 | +12.3% over scratch |
| Transfer early adaptation (+17.6%) | Ep 1–10 on target domain |
| Full System wins tight_budget env | 13.26 vs 12.78 (PPO Only) |
| Real agent execution | All 4 agents run `_execute()` every training step |

### Technical Contributions

- Hierarchical two-level RL: PPO (strategic) + LinUCB (tactical) with correct
  importance ratios
- Count-based intrinsic motivation via random-projection LSH for bandit exploration
- Real agent execution in the training loop: agents perform coverage-gap analysis,
  credibility scoring, synthesis, and contradiction detection during every step
- Inter-agent messaging: EvaluatorAgent credibility warnings propagate to SearchAgent
  domain avoidance in real time
- Transfer learning via 0.3× LR fine-tuning with architecture-matched 4-action PPO

The complete source code, experiment data, and results are available at:
https://github.com/UshakeShravya/madison-rl-intel
