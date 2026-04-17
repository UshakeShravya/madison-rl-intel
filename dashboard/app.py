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
    st.header("🌐 Live Demo — RL Policy + Real APIs")
    st.info(
        "The trained **PPO meta-controller** selects which agent to activate. "
        "The **contextual bandit** selects the real API tool to call. "
        "Real sources (Wikipedia, arXiv, OpenAlex) are retrieved, semantically "
        "ranked by the **SemanticRelevanceExtractor**, and assembled into a research report."
    )

    col_q, col_t = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Research query:",
            placeholder="e.g. What are the latest advances in quantum computing?",
        )
    with col_t:
        query_type_live = st.selectbox("Query type", [
            "exploratory", "technical", "factual", "comparative",
            "temporal", "causal", "opinion", "quantitative",
        ], key="live_qt")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_real = st.button("🔍 Run RL-Directed Research", type="primary")
    with col2:
        n_steps = st.slider("Max steps", 4, 12, 8, key="live_steps")
    with col3:
        show_raw = st.checkbox("Show raw API JSON")

    if run_real and query:
        from src.inference import LiveResearchSession

        session = LiveResearchSession()
        progress_bar = st.progress(0, text="Initialising RL policy...")

        step_placeholder = st.empty()
        step_rows = []

        # Patch research() to stream step updates into Streamlit
        original_research = session.research

        def _streaming_research(q, qt, ms, verbose):
            report = original_research(q, query_type=qt, max_steps=ms, verbose=False)
            return report

        with st.spinner("RL policy directing real API calls..."):
            report = session.research(
                query, query_type=query_type_live,
                max_steps=n_steps, verbose=False,
            )
        progress_bar.progress(1.0, text="Done!")

        # ── Summary metrics ───────────────────────────────────────────────────
        st.subheader("Research Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sources Found",   len(report.sources))
        c2.metric("Steps Taken",     report.steps)
        c3.metric("Est. Reward",     f"{report.total_reward:.2f}")
        c4.metric("Elapsed",         f"{report.elapsed_sec:.1f}s")
        avg_cov = round(float(np.mean(list(report.coverage.values()))) if report.coverage else 0, 2)
        c5.metric("Avg Coverage",    avg_cov)

        # ── Agent activation log ──────────────────────────────────────────────
        st.subheader("🤖 RL Policy — Agent & Tool Decisions")
        if report.agent_log:
            log_df = pd.DataFrame(report.agent_log)
            log_df["Compatible"] = log_df["compatible"].map({True: "✅", False: "❌"})
            st.dataframe(
                log_df[["step","agent","tool","Compatible","domain",
                         "credibility","latency","reward","coverage"]],
                use_container_width=True, hide_index=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                ac = log_df["agent"].value_counts()
                fig = px.pie(values=ac.values, names=ac.index,
                             title="Agent Activation (PPO decisions)",
                             color_discrete_sequence=["#3B8BD4","#1D9E75","#534AB7","#E8593C"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                tc = log_df["tool"].value_counts()
                fig = px.bar(x=tc.index, y=tc.values,
                             title="Tool Selection (Bandit decisions)",
                             color_discrete_sequence=["#3B8BD4"])
                fig.update_layout(xaxis_tickangle=-30, height=320)
                st.plotly_chart(fig, use_container_width=True)
            with col3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=log_df["step"], y=log_df["coverage"],
                    mode="lines+markers", name="Coverage",
                    line=dict(color="#1D9E75", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=log_df["step"], y=log_df["reward"].cumsum() / log_df["step"],
                    mode="lines", name="Avg Reward",
                    line=dict(color="#E8593C", width=2, dash="dot"),
                ))
                fig.update_layout(title="Coverage & Reward over Steps",
                                  xaxis_title="Step", height=320)
                st.plotly_chart(fig, use_container_width=True)

        # ── Subtopic coverage ─────────────────────────────────────────────────
        if report.coverage:
            st.subheader("📊 Subtopic Coverage (Semantic)")
            cov_df = pd.DataFrame(list(report.coverage.items()),
                                  columns=["Subtopic", "Coverage"])
            fig = px.bar(cov_df, x="Subtopic", y="Coverage",
                         color="Coverage", color_continuous_scale="Blues",
                         title="Semantic Coverage per Query Subtopic")
            fig.update_layout(yaxis_range=[0, 1], height=300)
            st.plotly_chart(fig, use_container_width=True)

        # ── Sources found ─────────────────────────────────────────────────────
        if report.sources:
            st.subheader("📚 Sources Retrieved")
            src_df = pd.DataFrame(report.sources)
            src_df = src_df.drop_duplicates(subset=["title","domain"]).sort_values("credibility", ascending=False)
            # Make URLs clickable
            src_df["url"] = src_df["url"].apply(
                lambda u: f'<a href="{u}" target="_blank">🔗 link</a>' if u else ""
            )
            st.write(
                src_df[["title","domain","source_type","credibility","url"]]
                .to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                dt = src_df["source_type"].value_counts()
                fig = px.pie(values=dt.values, names=dt.index,
                             title="Source Type Distribution",
                             color_discrete_sequence=["#3B8BD4","#1D9E75","#E8593C","#534AB7"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                dm = src_df["domain"].value_counts()
                fig = px.bar(x=dm.index, y=dm.values,
                             title="Domains Retrieved",
                             color_discrete_sequence=["#534AB7"])
                fig.update_layout(xaxis_tickangle=-30, height=320)
                st.plotly_chart(fig, use_container_width=True)

        # ── Key passages (SemanticRelevanceExtractor) ─────────────────────────
        if report.key_passages:
            st.subheader("📄 Key Passages — SemanticRelevanceExtractor")
            st.caption("Extracted using sentence-transformer embeddings (all-MiniLM-L6-v2), "
                       "not keyword matching — captures meaning-level relevance.")
            for i, passage in enumerate(report.key_passages[:4], 1):
                with st.expander(f"Passage {i}"):
                    st.write(passage[:400])

        if show_raw:
            st.subheader("Raw Report JSON")
            st.json({
                "query": report.query,
                "query_type": report.query_type,
                "steps": report.steps,
                "total_reward": report.total_reward,
                "coverage": report.coverage,
                "agent_log": report.agent_log,
                "sources": report.sources,
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