"""
Streamlit Dashboard for Madison RL Intelligence Agent.

Run with: streamlit run dashboard/app.py

Three tabs:
  1. Training Results — learning curves and comparison plots
  2. Live Demo — run a query and watch agents work
  3. Bandit Analysis — tool selection statistics
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.environment import ResearchEnv
from src.controller.ppo import PPOController
from src.bandits.contextual_bandit import MultiAgentBanditManager
from src.utils.config import MadisonConfig

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Madison RL Intelligence Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("Madison RL Intelligence Agent")
st.markdown("*Reinforcement learning-powered multi-agent research orchestrator*")


# ──────────────────────────────────────────────
# Tab 1: Training Results
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Training Results", "Live Demo", "Bandit Analysis"])

with tab1:
    st.header("Training Results")

    # Load training results if available
    results_path = Path("experiments/logs/training_results.json")
    experiment_path = Path("experiments/logs/experiment_results.json")

    if results_path.exists():
        with open(results_path) as f:
            training_data = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            # Learning curve
            rewards = training_data["episode_rewards"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=rewards, mode="lines", name="Raw",
                line=dict(color="rgba(59, 139, 212, 0.3)"),
            ))
            if len(rewards) > 20:
                window = 20
                smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
                fig.add_trace(go.Scatter(
                    x=list(range(window-1, window-1+len(smoothed))),
                    y=smoothed.tolist(), mode="lines", name="Smoothed",
                    line=dict(color="#3B8BD4", width=3),
                ))
            fig.update_layout(
                title="Training Reward Over Episodes",
                xaxis_title="Episode", yaxis_title="Total Reward",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Coverage and diversity from metrics history
            if "metrics_history" in training_data:
                metrics = training_data["metrics_history"]
                coverages = [m["coverage"] for m in metrics]
                diversities = [m["diversity"] for m in metrics]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=coverages, mode="lines", name="Coverage",
                    line=dict(color="#1D9E75"),
                ))
                fig.add_trace(go.Scatter(
                    y=diversities, mode="lines", name="Diversity",
                    line=dict(color="#E8593C"),
                ))
                fig.update_layout(
                    title="Coverage & Diversity Over Training",
                    xaxis_title="Episode", yaxis_title="Score",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.subheader("Summary Statistics")
        last_50 = rewards[-50:] if len(rewards) >= 50 else rewards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Reward (Last 50)", f"{np.mean(last_50):.2f}")
        col2.metric("Max Reward", f"{max(rewards):.2f}")
        col3.metric("Total Episodes", len(rewards))
        col4.metric("Std Dev (Last 50)", f"{np.std(last_50):.2f}")

    else:
        st.warning("No training results found. Run training first: `python -m src.train`")

    # Experiment comparison
    if experiment_path.exists():
        st.subheader("Comparative Analysis")
        with open(experiment_path) as f:
            exp_data = json.load(f)

        # Bar chart of final performance
        names = []
        means = []
        stds = []
        for name, seeds in exp_data.items():
            final_rewards = [np.mean(s[-50:]) if len(s) >= 50 else np.mean(s) for s in seeds]
            names.append(name)
            means.append(np.mean(final_rewards))
            stds.append(np.std(final_rewards))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names, y=means,
            error_y=dict(type="data", array=stds),
            marker_color=["#888888", "#E8593C", "#534AB7", "#1D9E75", "#3B8BD4"],
        ))
        fig.update_layout(
            title="Final Performance Comparison (Mean ± Std)",
            yaxis_title="Mean Reward",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 2: Live Demo
# ──────────────────────────────────────────────
with tab2:
    st.header("Live Demo — Watch Agents Work")

    query_type = st.selectbox(
        "Select query type:",
        ["factual", "comparative", "exploratory", "temporal",
         "causal", "technical", "opinion", "quantitative"],
    )

    if st.button("Run Research Episode", type="primary"):
        config = MadisonConfig()
        env = ResearchEnv(config.environment, config.reward)

        # Initialize RL components
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        ppo = PPOController(obs_dim, n_actions, config.ppo)
        bandits = MultiAgentBanditManager(config.bandit, {
            "search": ["web_scraper", "api_client", "academic_search"],
            "evaluator": ["credibility_scorer", "web_scraper"],
            "synthesis": ["relevance_extractor", "pdf_parser"],
            "deep_dive": ["web_scraper", "api_client", "relevance_extractor"],
        })

        # Try to load trained checkpoint
        checkpoint_dir = Path("experiments/checkpoints")
        checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        if checkpoints:
            ppo.load(str(checkpoints[-1]))
            st.success(f"Loaded trained model: {checkpoints[-1].name}")
        else:
            st.info("No trained model found — using untrained policy")

        # Run episode
        obs, info = env.reset()
        steps_data = []
        total_reward = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        done = False
        step = 0
        while not done:
            action, log_prob, value = ppo.select_action(obs)
            agent_idx = action // env.n_tools
            agent_name = env.AGENTS[agent_idx].value

            context = obs[:config.bandit.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)

            tool_names_list = [t.value for t in env.TOOLS]
            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = action % env.n_tools
            final_action = agent_idx * env.n_tools + tool_idx

            next_obs, reward, term, trunc, step_info = env.step(final_action)
            done = term or trunc
            total_reward += reward

            steps_data.append({
                "Step": step + 1,
                "Agent": agent_name,
                "Tool": tool_name,
                "Compatible": step_info["compatible"],
                "Relevance": round(step_info["relevance"], 3),
                "Reward": round(reward, 3),
                "Coverage": round(step_info["coverage"], 3),
                "Budget Used": round(step_info["budget_used"], 2),
            })

            step += 1
            progress_bar.progress(min(step / 20, 1.0))
            status_text.text(
                f"Step {step} | Agent: {agent_name} | Tool: {tool_name} | "
                f"Reward: {reward:.2f} | Coverage: {step_info['coverage']:.2f}"
            )
            obs = next_obs

        # Display results
        st.subheader("Episode Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reward", f"{total_reward:.2f}")
        col2.metric("Steps Taken", step)
        col3.metric("Final Coverage", f"{env.memory.get_coverage_score():.2f}")
        col4.metric("Source Diversity", f"{env.memory.get_source_diversity():.2f}")

        # Step-by-step table
        st.subheader("Step-by-Step Actions")
        df = pd.DataFrame(steps_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Agent usage pie chart
        col1, col2 = st.columns(2)
        with col1:
            agent_counts = df["Agent"].value_counts()
            fig = px.pie(
                values=agent_counts.values,
                names=agent_counts.index,
                title="Agent Activation Distribution",
                color_discrete_sequence=["#3B8BD4", "#1D9E75", "#534AB7", "#E8593C"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            tool_counts = df["Tool"].value_counts()
            fig = px.pie(
                values=tool_counts.values,
                names=tool_counts.index,
                title="Tool Selection Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 3: Bandit Analysis
# ──────────────────────────────────────────────
with tab3:
    st.header("Bandit Tool Selection Analysis")

    results_path = Path("experiments/logs/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

        if "bandit_stats" in data:
            stats = data["bandit_stats"]

            for agent_name, agent_stats in stats.items():
                st.subheader(f"Agent: {agent_name}")

                # Extract pull counts and avg rewards
                tools = []
                pulls = []
                avg_rewards = []
                for key, val in agent_stats.items():
                    if key.endswith("_pulls"):
                        tool = key.replace("_pulls", "")
                        tools.append(tool)
                        pulls.append(val)
                        reward_key = f"{tool}_avg_reward"
                        avg_rewards.append(agent_stats.get(reward_key, 0))

                if tools:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            x=tools, y=pulls,
                            title=f"Tool Pull Counts — {agent_name}",
                            labels={"x": "Tool", "y": "Times Selected"},
                            color_discrete_sequence=["#3B8BD4"],
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = px.bar(
                            x=tools, y=avg_rewards,
                            title=f"Average Reward per Tool — {agent_name}",
                            labels={"x": "Tool", "y": "Avg Reward"},
                            color_discrete_sequence=["#1D9E75"],
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training data found. Run training first.")

    st.markdown("---")
    st.markdown(
        "*Madison RL Intelligence Agent — "
        "Reinforcement Learning for Agentic AI Systems*"
    )