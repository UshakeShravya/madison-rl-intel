"""
Streamlit Dashboard for Madison RL Intelligence Agent.

Run with: PYTHONPATH=. streamlit run dashboard/app.py

Four tabs:
  1. Training Results — learning curves and comparison plots
  2. Live Demo (Simulation) — watch RL agents work in simulation
  3. Live Demo (Real APIs) — real queries hitting Wikipedia + Semantic Scholar
  4. Bandit Analysis — tool selection statistics
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Madison RL Intelligence Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Madison RL Intelligence Agent")
st.markdown("*Hierarchical reinforcement learning — PPO + contextual bandits + transfer learning*")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Training Results",
    "🤖 Live Demo (Simulation)",
    "🌐 Live Demo (Real APIs)",
    "🎰 Bandit Analysis"
])

# ── TAB 1: TRAINING RESULTS ──
with tab1:
    st.header("Training Results")

    results_path = Path("experiments/logs/training_results.json")
    experiment_path = Path("experiments/logs/experiment_results.json")
    transfer_path = Path("experiments/logs/transfer_results.json")

    if results_path.exists():
        with open(results_path) as f:
            training_data = json.load(f)

        rewards = training_data["episode_rewards"]

        col1, col2, col3, col4 = st.columns(4)
        last_50 = rewards[-50:] if len(rewards) >= 50 else rewards
        col1.metric("Mean Reward (Last 50)", f"{np.mean(last_50):.2f}")
        col2.metric("Max Reward", f"{max(rewards):.2f}")
        col3.metric("Total Episodes", len(rewards))
        col4.metric("Improvement vs Random", f"{((np.mean(last_50) / 7.1) - 1) * 100:.0f}%")

        st.subheader("Learning Curve")
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=rewards, mode="lines", name="Raw reward",
                line=dict(color="rgba(59,139,212,0.25)"),
            ))
            if len(rewards) > 20:
                smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")
                fig.add_trace(go.Scatter(
                    x=list(range(19, 19+len(smoothed))),
                    y=smoothed.tolist(), mode="lines", name="Smoothed (20-ep)",
                    line=dict(color="#3B8BD4", width=2.5),
                ))
            if training_data.get("eval_rewards"):
                fig.add_trace(go.Scatter(
                    x=training_data["eval_episodes"],
                    y=training_data["eval_rewards"],
                    mode="markers+lines", name="Eval",
                    line=dict(color="#E8593C", dash="dot"),
                    marker=dict(size=6),
                ))
            fig.update_layout(title="PPO Training Reward", xaxis_title="Episode",
                              yaxis_title="Total Reward", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "metrics_history" in training_data:
                metrics = training_data["metrics_history"]
                coverages = [m["coverage"] for m in metrics]
                diversities = [m["diversity"] for m in metrics]
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=coverages, mode="lines", name="Coverage",
                                         line=dict(color="#1D9E75")))
                fig.add_trace(go.Scatter(y=diversities, mode="lines", name="Diversity",
                                         line=dict(color="#E8593C")))
                fig.update_layout(title="Coverage & Diversity Over Training",
                                  xaxis_title="Episode", yaxis_title="Score", height=380)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training results found. Run: `python -m src.train`")

    if experiment_path.exists():
        st.subheader("Comparative Analysis — All 5 Configurations")
        with open(experiment_path) as f:
            exp_data = json.load(f)

        col1, col2 = st.columns(2)
        colors = ["#888888", "#E8593C", "#534AB7", "#1D9E75", "#3B8BD4"]

        with col1:
            names, means, stds = [], [], []
            for name, seeds in exp_data.items():
                final = [np.mean(s[-50:]) if len(s) >= 50 else np.mean(s) for s in seeds]
                names.append(name)
                means.append(np.mean(final))
                stds.append(np.std(final))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=means,
                error_y=dict(type="data", array=stds),
                marker_color=colors[:len(names)]))
            fig.update_layout(title="Final Performance (Mean ± Std)",
                              yaxis_title="Mean Reward", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            for i, (name, seeds) in enumerate(exp_data.items()):
                min_len = min(len(s) for s in seeds)
                mean = np.mean([s[:min_len] for s in seeds], axis=0)
                if len(mean) > 20:
                    smoothed = np.convolve(mean, np.ones(20)/20, mode="valid")
                    fig.add_trace(go.Scatter(
                        x=list(range(19, 19+len(smoothed))),
                        y=smoothed.tolist(), mode="lines", name=name,
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            fig.update_layout(title="Learning Curves Comparison",
                              xaxis_title="Episode", yaxis_title="Reward", height=400)
            st.plotly_chart(fig, use_container_width=True)

    if transfer_path.exists():
        st.subheader("Transfer Learning Results")
        with open(transfer_path) as f:
            tdata = json.load(f)

        col1, col2 = st.columns(2)
        tcolors = {"from_scratch": "#E8593C", "transfer": "#3B8BD4", "no_finetune": "#888888"}
        tlabels = {"from_scratch": "From scratch", "transfer": "Transfer learning", "no_finetune": "No fine-tuning"}

        with col1:
            fig = go.Figure()
            for key in ["from_scratch", "transfer", "no_finetune"]:
                if key not in tdata:
                    continue
                seeds = tdata[key]
                min_len = min(len(s) for s in seeds)
                mean = np.mean([s[:min_len] for s in seeds], axis=0)
                std = np.std([s[:min_len] for s in seeds], axis=0)
                fig.add_trace(go.Scatter(
                    y=mean.tolist(), mode="lines", name=tlabels[key],
                    line=dict(color=tcolors[key], width=2),
                ))
            fig.update_layout(title="Transfer Learning Adaptation Speed",
                              xaxis_title="Episodes on target domain",
                              yaxis_title="Reward", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            names2, means2, stds2 = [], [], []
            for key in ["from_scratch", "transfer", "no_finetune"]:
                if key not in tdata:
                    continue
                seeds = tdata[key]
                final = [np.mean(s[-20:]) for s in seeds]
                names2.append(tlabels[key])
                means2.append(np.mean(final))
                stds2.append(np.std(final))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=names2, y=means2,
                error_y=dict(type="data", array=stds2),
                marker_color=[tcolors[k] for k in ["from_scratch", "transfer", "no_finetune"]
                              if k in tdata],
            ))
            fig.update_layout(title="Final Performance on Target Domain",
                              yaxis_title="Mean Reward (Last 20 eps)", height=380)
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: SIMULATION DEMO ──
with tab2:
    st.header("Live Demo — Simulation Mode")
    st.info("Watch the trained RL policy allocate agents and select tools in real time.")

    query_type = st.selectbox("Select query type:", [
        "factual", "comparative", "exploratory", "temporal",
        "causal", "technical", "opinion", "quantitative"
    ])

    if st.button("▶ Run Simulation Episode", type="primary"):
        from src.environment import ResearchEnv
        from src.controller.ppo import PPOController
        from src.bandits.contextual_bandit import MultiAgentBanditManager
        from src.utils.config import MadisonConfig

        config = MadisonConfig()
        env = ResearchEnv(config.environment, config.reward)
        obs_dim = env.observation_space.shape[0]
        ppo = PPOController(obs_dim, env.action_space.n, config.ppo)
        bandits = MultiAgentBanditManager(config.bandit, {
            "search": ["web_scraper", "api_client", "academic_search"],
            "evaluator": ["credibility_scorer", "web_scraper"],
            "synthesis": ["relevance_extractor", "pdf_parser"],
            "deep_dive": ["web_scraper", "api_client", "relevance_extractor"],
        })

        checkpoint_dir = Path("experiments/checkpoints")
        checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        if checkpoints:
            ppo.load(str(checkpoints[-1]))
            st.success(f"✅ Loaded trained model: {checkpoints[-1].name}")
        else:
            st.info("ℹ️ No checkpoint found — using untrained policy")

        obs, info = env.reset()
        steps_data = []
        total_reward = 0.0
        done = False
        step = 0
        progress = st.progress(0)
        status = st.empty()

        while not done:
            action, log_prob, value = ppo.select_action(obs)
            agent_idx = action // env.n_tools
            agent_name = env.AGENTS[agent_idx].value
            context = obs[:config.bandit.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)
            tool_names_list = [t.value for t in env.TOOLS]
            tool_idx = tool_names_list.index(tool_name) if tool_name in tool_names_list else action % env.n_tools
            final_action = agent_idx * env.n_tools + tool_idx
            next_obs, reward, term, trunc, step_info = env.step(final_action)
            done = term or trunc
            total_reward += reward
            steps_data.append({
                "Step": step + 1,
                "Agent": agent_name,
                "Tool": tool_name,
                "Compatible": "✅" if step_info["compatible"] else "❌",
                "Relevance": round(step_info["relevance"], 3),
                "Reward": round(reward, 3),
                "Coverage": round(step_info["coverage"], 3),
                "Budget Used": round(step_info["budget_used"], 2),
            })
            step += 1
            progress.progress(min(step / 20, 1.0))
            status.text(f"Step {step} | Agent: {agent_name} | Tool: {tool_name} | Reward: {reward:.2f}")
            obs = next_obs

        st.subheader("Episode Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reward", f"{total_reward:.2f}")
        c2.metric("Steps", step)
        c3.metric("Coverage", f"{env.memory.get_coverage_score():.2f}")
        c4.metric("Diversity", f"{env.memory.get_source_diversity():.2f}")

        st.subheader("Step-by-Step Agent Decisions")
        st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        df = pd.DataFrame(steps_data)
        with col1:
            ac = df["Agent"].value_counts()
            fig = px.pie(values=ac.values, names=ac.index, title="Agent Activation",
                         color_discrete_sequence=["#3B8BD4","#1D9E75","#534AB7","#E8593C"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            tc = df["Tool"].value_counts()
            fig = px.pie(values=tc.values, names=tc.index, title="Tool Selection",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: REAL API DEMO ──
with tab3:
    st.header("🌐 Live Demo — Real Research (Wikipedia + Semantic Scholar)")
    st.info(
        "Type any real research question. The system queries Wikipedia and "
        "Semantic Scholar, then uses the trained RL policy to evaluate and rank sources."
    )

    query = st.text_input(
        "Research query:",
        placeholder="e.g. What are the latest advances in quantum computing?",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        run_real = st.button("🔍 Run Real Research", type="primary")
    with col2:
        show_raw = st.checkbox("Show raw API responses")

    if run_real and query:
        from src.tools.real_apis import RealResearchEngine
        from src.tools.relevance_extractor import RelevanceExtractor

        with st.spinner("Querying real sources — Wikipedia, Semantic Scholar, DuckDuckGo..."):
            engine = RealResearchEngine()
            results = engine.research(query)

        st.subheader("Research Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sources Found", results["total_sources"])
        c2.metric("Domains Covered", len(results["domains"]))
        c3.metric("APIs Queried", len(results["timeline"]))
        c4.metric("Query", query[:30] + "..." if len(query) > 30 else query)

        # API timeline
        st.subheader("API Performance")
        tl_df = pd.DataFrame(results["timeline"])
        if not tl_df.empty:
            fig = go.Figure(go.Bar(
                x=tl_df["api"], y=tl_df["time"],
                marker_color=["#3B8BD4", "#1D9E75", "#E8593C"],
                text=[f"{r} results" for r in tl_df["results"]],
                textposition="auto",
            ))
            fig.update_layout(title="Response Time per API (seconds)",
                              yaxis_title="Time (s)", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Wikipedia summary
        if results.get("wiki_summary"):
            st.subheader("📖 Wikipedia Summary")
            st.info(results["wiki_summary"][:600] + "..." if len(results["wiki_summary"]) > 600
                    else results["wiki_summary"])

        # Use RelevanceExtractor on real content
        if results["sources"]:
            st.subheader("🔍 Relevance-Ranked Sources (Custom Tool)")
            extractor = RelevanceExtractor()

            source_data = []
            for s in results["sources"]:
                source_data.append({
                    "Title": s.title[:70],
                    "Type": s.source_type,
                    "Domain": s.domain,
                    "Relevance": round(s.relevance_score, 3),
                    "Credibility": round(s.credibility_score, 3),
                    "Combined Score": round(s.relevance_score * 0.5 + s.credibility_score * 0.5, 3),
                    "URL": s.url,
                })

            df = pd.DataFrame(source_data).sort_values("Combined Score", ascending=False)
            st.dataframe(df.drop(columns=["URL"]), use_container_width=True, hide_index=True)

            # Source type breakdown
            col1, col2 = st.columns(2)
            with col1:
                type_counts = df["Type"].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                             title="Source Type Distribution",
                             color_discrete_sequence=["#3B8BD4", "#1D9E75", "#E8593C"])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["Relevance"], y=df["Credibility"],
                    mode="markers+text",
                    text=df["Domain"],
                    textposition="top center",
                    marker=dict(
                        size=df["Combined Score"] * 30 + 8,
                        color=df["Combined Score"],
                        colorscale="Viridis",
                        showscale=True,
                    )
                ))
                fig.update_layout(
                    title="Relevance vs Credibility (size = combined score)",
                    xaxis_title="Relevance Score",
                    yaxis_title="Credibility Score",
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Passage extraction on top source
            top_source = results["sources"][0]
            if top_source.snippet:
                st.subheader("📄 Relevant Passage Extraction (Custom NLP Tool)")
                st.markdown(f"**Source:** {top_source.title}")
                passages = extractor.extract(
                    top_source.snippet * 3,
                    query, top_k=3
                )
                for i, p in enumerate(passages):
                    st.markdown(f"**Passage {i+1}** (relevance: {p.relevance_score:.3f})")
                    st.text(p.text[:200])

        if show_raw:
            st.subheader("Raw API Response")
            st.json({
                "query": results["query"],
                "total_sources": results["total_sources"],
                "domains": results["domains"],
                "sources": [
                    {"title": s.title, "url": s.url,
                     "relevance": s.relevance_score, "credibility": s.credibility_score}
                    for s in results["sources"]
                ]
            })

# ── TAB 4: BANDIT ANALYSIS ──
with tab4:
    st.header("Bandit Tool Selection Analysis")
    results_path = Path("experiments/logs/training_results.json")

    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

        if "bandit_stats" in data:
            stats = data["bandit_stats"]
            for agent_name, agent_stats in stats.items():
                st.subheader(f"Agent: {agent_name}")
                tools, pulls, avg_rewards = [], [], []
                for key, val in agent_stats.items():
                    if key.endswith("_pulls"):
                        tool = key.replace("_pulls", "")
                        tools.append(tool)
                        pulls.append(val)
                        avg_rewards.append(agent_stats.get(f"{tool}_avg_reward", 0))

                if tools:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(x=tools, y=pulls,
                                     title=f"Pull counts — {agent_name}",
                                     color_discrete_sequence=["#3B8BD4"])
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.bar(x=tools, y=avg_rewards,
                                     title=f"Avg reward per tool — {agent_name}",
                                     color_discrete_sequence=["#1D9E75"])
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training data found. Run training first.")

    st.markdown("---")
    st.markdown("*Madison RL Intelligence Agent — Reinforcement Learning for Agentic AI Systems*")
    st.markdown("*GitHub: https://github.com/UshakeShravya/madison-rl-intel*")