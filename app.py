"""
Streamlit Web App: AI Debate Arena
==================================
Run the Multi-Agent Debate in your browser powered by LangGraph, LangChain, and Ollama.
Launch with:
    streamlit run app.py
"""

import streamlit as st
from langchain_ollama import ChatOllama
import agent

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Debate Arena",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .pro-box {
        background-color: rgba(46, 125, 50, 0.08);
        border-left: 5px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .con-box {
        background-color: rgba(198, 40, 40, 0.08);
        border-left: 5px solid #c62828;
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .judge-box {
        background-color: rgba(25, 118, 210, 0.08);
        border: 2px solid #1976d2;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .winner-badge-pro {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .winner-badge-con {
        display: inline-block;
        background-color: #c62828;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .winner-badge-tie {
        display: inline-block;
        background-color: #f57c00;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar Settings
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Debate Settings")

    model_name = st.text_input(
        "Ollama Model",
        value=agent.MODEL_NAME,
        help="Make sure this model is pulled in Ollama (e.g. llama3.2).",
    )

    rounds = st.slider(
        "Number of Rounds",
        min_value=1,
        max_value=4,
        value=agent.ROUNDS,
        step=1,
        help="Each round includes 1 speech from PRO and 1 speech from CON.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=agent.TEMPERATURE,
        step=0.05,
        help="Higher values make speeches more creative and expressive.",
    )

    st.markdown("---")
    st.subheader("💡 Suggested Motions")
    preset_motions = [
        "Remote work is better than office work",
        "Artificial Intelligence will create more jobs than it destroys",
        "Social media does more harm than good to society",
        "Space exploration is worth the cost",
        "Universal Basic Income is essential in the automated era",
    ]

    selected_preset = st.selectbox(
        "Choose an example motion:",
        options=["(Custom)"] + preset_motions,
    )

    st.markdown("---")
    with st.expander("ℹ️ How it Works"):
        st.markdown(
            """
            - **PRO Agent**: Argues strictly **FOR** the motion (max 50 words).
            - **CON Agent**: Argues strictly **AGAINST** the motion and answers opponent's last point.
            - **StateGraph**: Cycles `PRO -> CON` for the chosen rounds.
            - **JUDGE Agent**: Impartial evaluation declaring the winner & reason.
            """
        )

# ==============================================================================
# Header
# ==============================================================================
st.markdown("<div class='main-title'>🎙️ Autonomous AI Debate Arena</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Watch two AI agents debate any topic in real-time under strict word limits, adjudicated by an AI judge.</div>",
    unsafe_allow_html=True,
)

# Default motion value logic
default_input_value = ""
if selected_preset != "(Custom)":
    default_input_value = selected_preset

col1, col2 = st.columns([5, 1])
with col1:
    motion = st.text_input(
        "Enter Motion / Topic to Debate:",
        value=default_input_value,
        placeholder="e.g. Remote work is better than office work",
        label_visibility="collapsed",
    )
with col2:
    start_debate = st.button("🚀 Start Debate", type="primary", use_container_width=True)

# Session state initialization
if "debate_history" not in st.session_state:
    st.session_state.debate_history = None

# ==============================================================================
# Execution & Live Streaming
# ==============================================================================
if start_debate:
    if not motion.strip():
        st.warning("⚠️ Please enter a motion or select one from the sidebar.")
    else:
        st.session_state.debate_history = None
        motion_clean = motion.strip()

        st.subheader(f"📢 Motion: *\"{motion_clean}\"*")

        speeches_container = st.container()
        verdict_container = st.container()

        speeches_record = []
        final_verdict = None
        final_winner = None
        current_round = 1

        try:
            with st.spinner(f"Initializing LangGraph StateGraph with Ollama ({model_name})..."):
                llm = ChatOllama(model=model_name, temperature=temperature)
                graph = agent.build_debate_graph(llm=llm, max_rounds=rounds)

            initial_state: agent.DebateState = {
                "motion": motion_clean,
                "speeches": [],
                "round": 0,
                "verdict": None,
                "winner": None,
            }

            status_indicator = st.status(f"⚡ Debate in progress (Round {current_round}/{rounds})...", expanded=True)

            for event in graph.stream(initial_state, stream_mode="updates"):
                if "pro" in event:
                    raw_speech = event["pro"]["speeches"][-1]
                    clean_content = raw_speech[len("PRO: "):] if raw_speech.startswith("PRO: ") else raw_speech
                    speeches_record.append({"role": "PRO", "round": current_round, "text": clean_content})

                    with speeches_container:
                        st.markdown(
                            f"""
                            <div class='pro-box'>
                                <strong>🟢 PRO — Round {current_round}</strong><br>
                                {clean_content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    status_indicator.update(label=f"🔴 CON Debater is formulating rebuttal for Round {current_round}...")

                elif "con" in event:
                    raw_speech = event["con"]["speeches"][-1]
                    clean_content = raw_speech[len("CON: "):] if raw_speech.startswith("CON: ") else raw_speech
                    speeches_record.append({"role": "CON", "round": current_round, "text": clean_content})

                    with speeches_container:
                        st.markdown(
                            f"""
                            <div class='con-box'>
                                <strong>🔴 CON — Round {current_round}</strong><br>
                                {clean_content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    current_round += 1
                    if current_round <= rounds:
                        status_indicator.update(label=f"🟢 PRO Debater is speaking for Round {current_round}...")
                    else:
                        status_indicator.update(label="⚖️ Judge is analyzing the transcript...")

                elif "judge" in event:
                    final_verdict = event["judge"]["verdict"]
                    final_winner = event["judge"]["winner"]
                    status_indicator.update(label="✅ Debate completed!", state="complete", expanded=False)

                    with verdict_container:
                        badge_class = (
                            "winner-badge-pro"
                            if final_winner == "PRO"
                            else ("winner-badge-con" if final_winner == "CON" else "winner-badge-tie")
                        )
                        badge_label = f"🟢 PRO" if final_winner == "PRO" else (f"🔴 CON" if final_winner == "CON" else f"🤝 TIE")

                        st.markdown(
                            f"""
                            <div class='judge-box'>
                                <h3>⚖️ Official Judge Verdict</h3>
                                <p style='font-size: 1.15rem; margin-top: 0.5rem;'>
                                    <strong>Winner:</strong> <span class='{badge_class}'>{badge_label}</span>
                                </p>
                                <div style='white-space: pre-wrap; font-size: 1rem; margin-top: 0.8rem;'>
                                    {final_verdict}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Save in session state
            st.session_state.debate_history = {
                "motion": motion_clean,
                "speeches": speeches_record,
                "verdict": final_verdict,
                "winner": final_winner,
            }

        except Exception as err:
            st.error(f"❌ Error during debate execution: {err}")
            st.info(
                "💡 **Troubleshooting Steps:**\n"
                "1. Make sure Ollama is running in background: run `ollama serve`\n"
                f"2. Make sure the model is downloaded: run `ollama pull {model_name}`\n"
                "3. Verify that port `11434` is accessible locally."
            )

# Re-render from session state if already completed and page reloaded
elif st.session_state.debate_history:
    hist = st.session_state.debate_history
    st.subheader(f"📢 Motion: *\"{hist['motion']}\"*")

    for item in hist["speeches"]:
        if item["role"] == "PRO":
            st.markdown(
                f"""
                <div class='pro-box'>
                    <strong>🟢 PRO — Round {item['round']}</strong><br>
                    {item['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='con-box'>
                    <strong>🔴 CON — Round {item['round']}</strong><br>
                    {item['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if hist["verdict"]:
        winner = hist["winner"]
        badge_class = (
            "winner-badge-pro"
            if winner == "PRO"
            else ("winner-badge-con" if winner == "CON" else "winner-badge-tie")
        )
        badge_label = f"🟢 PRO" if winner == "PRO" else (f"🔴 CON" if winner == "CON" else f"🤝 TIE")

        st.markdown(
            f"""
            <div class='judge-box'>
                <h3>⚖️ Official Judge Verdict</h3>
                <p style='font-size: 1.15rem; margin-top: 0.5rem;'>
                    <strong>Winner:</strong> <span class='{badge_class}'>{badge_label}</span>
                </p>
                <div style='white-space: pre-wrap; font-size: 1rem; margin-top: 0.8rem;'>
                    {hist['verdict']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
