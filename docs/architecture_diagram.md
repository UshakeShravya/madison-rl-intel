# Madison RL Intelligence Agent — Architecture Diagram

## Two-Level Hierarchical Control

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                            RESEARCH QUERY                                   ║
║            text + subtopics + budget + query_type (8 categories)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      │  396-dim observation
                                      │  [query_emb(384) | coverage(8) | scalars(4)]
                                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PPO META-CONTROLLER (Strategic)                      ║
║                                                                              ║
║   Input:  396-dim state                                                      ║
║                                                                              ║
║   ┌──────────────────────────────────────────────────────────────────────┐   ║
║   │  Shared Backbone                                                     │   ║
║   │  Linear(396→256) → ReLU → Linear(256→256) → ReLU                   │   ║
║   └────────────────────────┬──────────────────────┬───────────────────┘   ║
║                             │                      │                         ║
║                    ┌────────▼────────┐    ┌────────▼────────┐               ║
║                    │  Policy Head    │    │   Value Head    │               ║
║                    │ Linear(256→4)   │    │  Linear(256→1)  │               ║
║                    │ + Softmax       │    │                 │               ║
║                    └────────┬────────┘    └────────┬────────┘               ║
║                             │                      │                         ║
║              agent_idx ∈ {0,1,2,3}            V(s_t)                        ║
║              log π(agent_idx|s_t)                                            ║
║                                                                              ║
║   Stored in rollout buffer: (s_t, agent_idx, r_t, done, log_prob, V(s_t))  ║
║   Update: PPO-clip objective over 4-class action space                      ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │
                          agent_idx ∈ {0,1,2,3}
                                    │
          ┌─────────────────────────┴──────────────────────────┐
          │                                                     │
          ▼                                                     │
╔═════════════════════════════════════════════════════╗        │
║           AGENT SELECTION (4 roles)                 ║        │
║                                                     ║        │
║  0: SearchAgent    → finds new sources              ║        │
║  1: EvaluatorAgent → scores source credibility      ║        │
║  2: SynthesisAgent → combines existing findings     ║        │
║  3: DeepDiveAgent  → fills coverage gaps            ║        ║
╚═════════════════════════════════════════════════════╝        │
          │                                                     │
          │  agent_name (string)                                │
          ▼                                                     │
╔══════════════════════════════════════════════════════════════╗│
║         PER-AGENT CONTEXTUAL BANDIT  (Tactical)             ║│
║                                                              ║│
║  4 independent bandits — one per agent role                  ║│
║  Context: first 128 dims of observation vector               ║│
║                                                              ║│
║  ┌─────────────────────────────────────────────────────┐    ║│
║  │ Selection score for arm i:                          │    ║│
║  │                                                     │    ║│
║  │  score_i = UCB_i(ctx) + novelty_i(ctx)              │    ║│
║  │                                                     │    ║│
║  │  UCB_i = θ_i^T ctx + α √(ctx^T A_i^{-1} ctx)      │    ║│
║  │                                                     │    ║│
║  │  novelty_i = β / √(count(arm_i, bucket(ctx)) + 1)  │    ║│
║  │              bucket via random projection P∈R^{8×128}│   ║│
║  └─────────────────────────────────────────────────────┘    ║│
║                                                              ║│
║  Tools per agent:                                            ║│
║  ┌──────────────┬──────────────────────────────────────┐    ║│
║  │ Search       │ web_scraper · api_client · acad_search│    ║│
║  │ Evaluator    │ credibility_scorer · web_scraper      │    ║│
║  │ Synthesis    │ relevance_extractor · pdf_parser      │    ║│
║  │ DeepDive     │ web_scraper · api_client · rel_extr   │    ║│
║  └──────────────┴──────────────────────────────────────┘    ║│
║                                                              ║│
║  Update: Online ridge regression (Sherman-Morrison)          ║│
║  Bandit only receives reward for its own tool pull           ║│
╚══════════════════════════════════════╤═══════════════════════╝│
                                       │                         │
                                  tool_idx ∈ {0..5}             │
                                       │                         │
                          ┌────────────┘                         │
                          │                                      │
                          │  final_action = agent_idx×6 + tool_idx
                          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                       RESEARCH ENVIRONMENT (Gym)                            ║
║                                                                              ║
║  ResearchEnv.step(final_action):                                             ║
║                                                                              ║
║  ① Decode: agent_idx = action // 6,  tool_idx = action % 6                 ║
║                                                                              ║
║  ② Compatibility check:                                                     ║
║     compatible = tool ∈ AGENT_TOOL_COMPATIBILITY[agent_role]               ║
║                                                                              ║
║  ③ Simulate latency: L ~ N(μ_L, σ_L)                                       ║
║     default: μ_L=0.5, σ_L=0.2  |  tight_budget: μ_L=1.0, σ_L=0.3         ║
║                                                                              ║
║  ④ Run real agent logic:                                                    ║
║     agent_result = agent_instance.process(memory, tool)                     ║
║     └─ SearchAgent:    coverage-gap analysis → target_subtopic              ║
║     └─ EvaluatorAgent: score unscored sources → broadcast warnings          ║
║     └─ SynthesisAgent: identify contradictions → list for DeepDive          ║
║     └─ DeepDiveAgent:  resolve contradictions or fill weakest gap           ║
║                                                                              ║
║  ⑤ Compute relevance (compatible path):                                    ║
║     relevance = clip(0.6 × affinity[query_type][tool]                       ║
║                    + 0.4 × agent_confidence                                 ║
║                    + ε,  0, 1)         ε ~ N(0, noise_std)                 ║
║     default noise_std=0.1  |  complex_queries: noise_std=0.25               ║
║                                                                              ║
║  ⑥ Retrieve source (credibility_score=0.0, EvaluatorAgent scores later)    ║
║     Finding.subtopic = agent_result["target_subtopic"]  (not random)       ║
║                                                                              ║
║  ⑦ Reward:                                                                  ║
║     R = 1.0·rel + 0.5·cov − 0.3·lat + 0.4·div + 0.6·(rel×cov)            ║
║                                                                              ║
║  ⑧ Termination:                                                             ║
║     budget_exceeded OR coverage>0.9  → terminated=True                     ║
║     step_count ≥ max_steps           → truncated=True                       ║
║                                                                              ║
╚══════════════════════════════════════╤═══════════════════════════════════════╝
                                       │
                           next_obs, reward, done
                                       │
          ┌────────────────────────────┘
          │
          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                       SHARED MEMORY (Inter-Agent Communication)             ║
║                                                                              ║
║  SharedMemory holds:                                                         ║
║    findings:     List[Finding]           (coverage_map updated each add)    ║
║    sources:      List[Source]            (credibility_score filled by eval) ║
║    coverage_map: Dict[subtopic → float]  (diminishing-returns accumulation) ║
║    messages:     List[AgentMessage]      (inter-agent broadcast channel)    ║
║    budget_used:  float                                                       ║
║                                                                              ║
║  Message flow per episode:                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │ EvaluatorAgent  ──credibility_warning──►  SearchAgent              │   ║
║  │   (domain, score < 0.5)                    (avoided_domains list)  │   ║
║  │                                                                      │   ║
║  │ SynthesisAgent  ──contradictions_found──►  DeepDiveAgent            │   ║
║  │   (list of topics with std(conf)>0.3)      (prioritised target)    │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
          │
          │  Updated observation
          ▼
   Back to PPO Meta-Controller (next step)
```

---

## Action Space Decomposition

```
Environment Action Space: Discrete(24)
action = agent_idx × 6 + tool_idx

         tool_idx
          0            1             2               3               4                5
     ─────────────────────────────────────────────────────────────────────────────────
  0  │web_scraper │api_client  │acad_search  │pdf_parser   │rel_extractor│cred_scorer│ SearchAgent
     │ compatible │compatible  │compatible   │INCOMPAT     │INCOMPAT     │INCOMPAT   │
  1  │web_scraper │api_client  │acad_search  │pdf_parser   │rel_extractor│cred_scorer│ EvaluatorAgent
a    │ compatible │INCOMPAT    │INCOMPAT     │INCOMPAT     │INCOMPAT     │compatible │
g    │────────────────────────────────────────────────────────────────────────────────│
e  2 │web_scraper │api_client  │acad_search  │pdf_parser   │rel_extractor│cred_scorer│ SynthesisAgent
n    │ INCOMPAT   │INCOMPAT    │INCOMPAT     │compatible   │compatible   │INCOMPAT   │
t    │────────────────────────────────────────────────────────────────────────────────│
_  3 │web_scraper │api_client  │acad_search  │pdf_parser   │rel_extractor│cred_scorer│ DeepDiveAgent
i    │ compatible │compatible  │INCOMPAT     │INCOMPAT     │compatible   │INCOMPAT   │
d    └─────────────────────────────────────────────────────────────────────────────────

PPO sees only: {0, 1, 2, 3}  (agent_idx row)
Bandit sees only: {0,1,…,5}  (tool_idx column, per-agent subset)
```

---

## Learning Update Cycle

```
Episode rollout (one episode = up to 20 steps):

  Step 1  →  Step 2  →  …  →  Step T (terminal)
    │           │                  │
    │           │                  │
    ▼           ▼                  ▼
  (s,a,r,d)  (s,a,r,d)  …   (s,a,r,d)      a = agent_idx ∈ {0..3}
    │                                        ← stored in PPO RolloutBuffer
    │
    ▼ (every bandit update is online, per step)
  bandit.update(agent_name, tool_arm, context, reward)
    └─ A_i += outer(x,x),  b_i += r·x
    └─ A_inv updated via Sherman-Morrison O(d²)
    └─ novelty.update(arm, context): count(bucket)++

After enough steps in buffer (≥ batch_size=64):
  ┌──────────────────────────────────────────┐
  │ PPO Update (4 epochs, minibatch size 64) │
  │                                          │
  │ For each minibatch:                      │
  │  1. Compute GAE advantages               │
  │     δ_t = r_t + γV(s_{t+1}) - V(s_t)   │
  │     Â_t = Σ (γλ)^k δ_{t+k}             │
  │                                          │
  │  2. Policy loss (clipped surrogate)      │
  │     r_t = π_new(a|s) / π_old(a|s)       │
  │     L_clip = min(r·Â, clip(r,1±ε)·Â)   │
  │                                          │
  │  3. Value loss                           │
  │     L_vf = (V(s) - (Â + V_old(s)))²    │
  │                                          │
  │  4. Combined + entropy bonus             │
  │     L = -L_clip + 0.5·L_vf - 0.01·H    │
  │                                          │
  │  5. Adam step, gradient clip at 0.5     │
  └──────────────────────────────────────────┘
```

---

## Transfer Learning Protocol

```
Source Domain (technical queries)          Target Domain (opinion queries)
─────────────────────────────────          ─────────────────────────────
PPO_source  ──300 episodes──►  θ_source
BanditsSearchSource  ─────────────────────────────────────────  (discarded)

                    ┌─────────────────────────┐
                    │ deep_copy(θ_source)     │
                    │         ↓               │
                    │   θ_transfer_init       │
                    │         ↓               │
                    │  LR ×= 0.3              │
                    │  (9×10⁻⁵ vs 3×10⁻⁴)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Fine-tune on target     │
                    │  (100 episodes)          │
                    │  Fresh bandits           │
                    │  (re-learn tool prefs)   │
                    └────────────┬────────────┘
                                 │
                         θ_transfer (final)

Evaluation (30 episodes each):
  Transfer:    θ_transfer evaluated on opinion
  From Scratch: θ_random  trained K ep on opinion
  No fine-tune: θ_source  evaluated on opinion (no update)

Results (K=1, 3 seeds):
  Transfer    25.49 ± 1.24  ← uses pre-trained agent coordination
  No fine-tune 26.51 ± 0.19  ← source policy generalises reasonably
  From Scratch 22.69 ± 0.16  ← very limited learning in K=1 steps
```

---

## Novelty Bonus Mechanics

```
Context x ∈ ℝ^128
              │
              │  Fixed projection matrix P ∈ ℝ^{8×128}  (seeded, never updated)
              ▼
       bits = sign(Px) ∈ {0,1}^8
              │
              │  pack_bits → integer in [0, 255]
              ▼
       bucket_key = (arm_idx, bucket_int)     ← 4×256 = 1024 possible keys

       count = visits[bucket_key]            default = 0

       bonus = β / √(count + 1)
             = 0.1 / √(count + 1)

  count=0  → bonus = 0.100  (never seen)
  count=3  → bonus = 0.050
  count=8  → bonus = 0.033
  count=24 → bonus = 0.020
  count=99 → bonus = 0.010  (approaches 0 asymptotically)

  Combined selection score:
  score_i = UCB_i(x) + bonus_i(x)
  arm* = argmax_i score_i
  After selection: visits[bucket_key]++
```
