# Madison RL Intelligence Agent

> A reinforcement learning-powered multi-agent research orchestrator that learns optimal strategies for information gathering, evaluation, and synthesis.

## Overview

Madison RL Intelligence Agent uses a hierarchical reinforcement learning architecture to coordinate specialized research agents. The system implements three RL methods operating at different decision levels:

1. PPO Meta-Controller (Policy Gradients) — Strategic decisions: which agents to activate and how to allocate budget
2. Contextual Bandits (LinUCB) — Tactical decisions: which tool each agent should use given the current context
3. Transfer Learning — Adaptation: policies trained on one query domain transfer efficiently to new domains

## Key Results

| Configuration | Mean Reward | vs Random Baseline |
|---|---|---|
| Random Baseline | ~7.1 | - |
| Heuristic | ~10.8 | +52% |
| Bandit Only | ~21.8 | +207% |
| Full System (PPO+Bandit) | ~22.5 | +217% |
| PPO Only (500 eps) | ~27.7 | +290% |

Transfer learning achieves peak performance from the first episodes on a new domain, while training from scratch requires a ramp-up period.

## Project Structure

    madison-rl-intel/
    ├── src/
    │   ├── agents/           # SearchAgent, EvaluatorAgent, SynthesisAgent, DeepDiveAgent
    │   ├── controller/       # PPO meta-controller + transfer learning
    │   ├── bandits/          # LinUCB contextual bandits for tool selection
    │   ├── tools/            # Custom NLP relevance extractor
    │   ├── environment/      # Gym-compatible research simulation
    │   ├── rewards/          # Multi-component reward engine
    │   └── utils/            # Config, data structures, logging
    ├── experiments/          # Configs, logs, checkpoints, result plots
    ├── dashboard/            # Streamlit interactive dashboard
    ├── tests/                # 20 unit tests (all passing)
    ├── docs/                 # Technical report
    └── demo/                 # Demo video materials

## Quick Start

    git clone https://github.com/YOUR_USERNAME/madison-rl-intel.git
    cd madison-rl-intel
    conda create -n madison python=3.11 -y
    conda activate madison
    pip install -r requirements.txt

    # Train the system
    python -m src.train

    # Run comparative experiments
    make experiment

    # Launch interactive dashboard
    make dashboard

    # Run tests
    pytest tests/ -v

## RL Methods Implemented

### 1. PPO Meta-Controller (Policy Gradients)
- Actor-critic architecture with shared backbone
- Generalized Advantage Estimation (GAE, lambda=0.95)
- Clipped surrogate objective (epsilon=0.2)
- Entropy bonus for exploration

### 2. Contextual Bandits (LinUCB)
- One bandit per agent for tool selection
- Upper Confidence Bound exploration
- Sherman-Morrison incremental inverse updates
- Context-aware: adapts tool choice to query type

### 3. Transfer Learning
- Pre-train on source domain (e.g. technical queries)
- Fine-tune on target domain with reduced learning rate (0.3x)
- Demonstrates faster adaptation vs training from scratch

## Custom Tool: Relevance Extractor
TF-IDF + cosine similarity based passage extraction tool that identifies the most relevant sections of documents given a research query. Used by SynthesisAgent and DeepDiveAgent.

## Reward Engineering
Multi-component reward function: R = alpha * relevance + beta * coverage - gamma * latency + delta * diversity + epsilon * quality

## Technologies
Python 3.11, PyTorch, Gymnasium, scikit-learn, Streamlit, Plotly, Pydantic, Loguru, pytest

## License
MIT