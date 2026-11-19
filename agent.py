"""
Debate Multi-Agent CLI
======================
A multi-agent debate arena built with LangChain, LangGraph, and Ollama (llama3.2).
A PRO debater and a CON debater alternate for a specified number of rounds (ROUNDS = 2),
then a neutral JUDGE evaluates the transcript and declares a winner.
"""

import os
import re
import sys
from typing import List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

# ==============================================================================
# 1. CONSTANTS & SYSTEM PROMPTS
# ==============================================================================

ROUNDS: int = 2
MODEL_NAME: str = os.getenv("OLLAMA_MODEL", "llama3.2")
TEMPERATURE: float = 0.7

PRO_PROMPT = """You are the PRO debater. You ARGUE FOR the motion — never switch sides.
Speak in max 50 words. If the opponent already spoke, answer their last point first.
Return ONLY your speech."""

CON_PROMPT = """You are the CON debater. You ARGUE AGAINST the motion — never switch sides.
Speak in max 50 words. Answer the opponent's last point first.
Return ONLY your speech."""

JUDGE_PROMPT = """You are a neutral JUDGE. Read the debate. Decide who argued better
(clear reasons, answered the opponent). Reply in EXACTLY this format:
Winner: PRO   (or)   Winner: CON
Reason: <one short sentence>"""


# ==============================================================================
# 2. STATE DEFINITION
# ==============================================================================

class DebateState(TypedDict):
    motion: str
    speeches: List[str]  # list of 'PRO: ...' / 'CON: ...'
    round: int
    verdict: Optional[str]
    winner: Optional[str]


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def trim_words(text: str, n: int = 50) -> str:
    """Trim text to at most n words."""
    words = text.strip().split()
    if len(words) <= n:
        return text.strip()
    return " ".join(words[:n])


def format_transcript(speeches: List[str]) -> str:
    """Format the debate speeches into a readable transcript string."""
    if not speeches:
        return "No speeches yet."
    return "\n\n".join(speeches)


def parse_winner(text: str) -> str:
    """Parse the judge's verdict to determine if Winner is PRO, CON, or TIE."""
    # Look for exact pattern 'Winner: PRO', 'Winner: CON', or 'Winner: TIE'
    match = re.search(r"Winner:\s*[*_]*(PRO|CON|TIE)[*_]*", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback heuristic if format slightly deviates
    upper = text.upper()
    has_pro = "PRO" in upper
    has_con = "CON" in upper
    if has_pro and not has_con:
        return "PRO"
    if has_con and not has_pro:
        return "CON"
    return "TIE"


def _clean_speech_output(raw_text: str, role_prefix: str, max_words: int = 50) -> str:
    """Strip leading quotes and redundant role labels before trimming."""
    text = raw_text.strip()
    # Strip wrapping quotes if LLM added them
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Strip redundant role prefix if the LLM outputted it
    prefix_pattern = rf"^(?:{role_prefix}|PRO|CON)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE).strip()

    return trim_words(text, max_words)


# ==============================================================================
# 4. NODES
# ==============================================================================

def pro_node(state: DebateState, llm: ChatOllama) -> dict:
    """
    PRO node: Argues FOR the motion.
    Reminded of its side every turn; speaks in max 50 words; answers opponent's last point;
    appends to speeches.
    """
    motion = state["motion"]
    speeches = state["speeches"]

    if not speeches:
        history_context = "No previous speeches. You open the debate."
    else:
        history_context = f"Debate so far:\n{format_transcript(speeches)}"

    human_content = (
        f"Motion: {motion}\n\n"
        f"{history_context}\n\n"
        "Remember: You are the PRO debater. You ARGUE FOR the motion — never switch sides.\n"
        "Speak in max 50 words. If the opponent already spoke, answer their last point first.\n"
        "Return ONLY your speech."
    )

    messages = [
        SystemMessage(content=PRO_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    cleaned_speech = _clean_speech_output(response.content, role_prefix="PRO", max_words=50)
    formatted_speech = f"PRO: {cleaned_speech}"

    return {
        "speeches": speeches + [formatted_speech],
    }


def con_node(state: DebateState, llm: ChatOllama) -> dict:
    """
    CON node: Argues AGAINST the motion.
    Reminded of its side every turn; speaks in max 50 words; answers opponent's last point first;
    appends to speeches AND increments round += 1.
    """
    motion = state["motion"]
    speeches = state["speeches"]

    history_context = f"Debate so far:\n{format_transcript(speeches)}"

    human_content = (
        f"Motion: {motion}\n\n"
        f"{history_context}\n\n"
        "Remember: You are the CON debater. You ARGUE AGAINST the motion — never switch sides.\n"
        "Speak in max 50 words. Answer the opponent's last point first.\n"
        "Return ONLY your speech."
    )

    messages = [
        SystemMessage(content=CON_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    cleaned_speech = _clean_speech_output(response.content, role_prefix="CON", max_words=50)
    formatted_speech = f"CON: {cleaned_speech}"

    return {
        "speeches": speeches + [formatted_speech],
        "round": state["round"] + 1,
    }


def judge_node(state: DebateState, llm: ChatOllama) -> dict:
    """
    JUDGE node: Reads the whole transcript;
    replies exactly 'Winner: PRO|CON' + 'Reason: <one sentence>'.
    """
    motion = state["motion"]
    speeches = state["speeches"]

    human_content = (
        f"Motion: {motion}\n\n"
        f"Debate Transcript:\n{format_transcript(speeches)}\n\n"
        "Decide who argued better (clear reasons, answered the opponent).\n"
        "Reply in EXACTLY this format:\n"
        "Winner: PRO   (or)   Winner: CON\n"
        "Reason: <one short sentence>"
    )

    messages = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    verdict = response.content.strip()
    winner = parse_winner(verdict)

    return {
        "verdict": verdict,
        "winner": winner,
    }


# ==============================================================================
# 5. GRAPH ROUTING & BUILDER
# ==============================================================================

def should_continue(state: DebateState) -> str:
    """
    Conditional edge router:
    round < ROUNDS ? pro : judge
    """
    if state["round"] < ROUNDS:
        return "pro"
    return "judge"


def build_debate_graph(llm: Optional[ChatOllama] = None, max_rounds: int = ROUNDS):
    """
    Builds and compiles the LangGraph StateGraph:
    START -> pro -> con -> conditional(round < max_rounds ? pro : judge) ; judge -> END
    """
    if llm is None:
        llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

    def route_condition(state: DebateState) -> str:
        if state["round"] < max_rounds:
            return "pro"
        return "judge"

    workflow = StateGraph(DebateState)

    # Add Nodes
    workflow.add_node("pro", lambda state: pro_node(state, llm))
    workflow.add_node("con", lambda state: con_node(state, llm))
    workflow.add_node("judge", lambda state: judge_node(state, llm))

    # Add Edges
    workflow.add_edge(START, "pro")
    workflow.add_edge("pro", "con")
    workflow.add_conditional_edges(
        "con",
        route_condition,
        {
            "pro": "pro",
            "judge": "judge",
        },
    )
    workflow.add_edge("judge", END)

    return workflow.compile()


# ==============================================================================
# 6. CLI EXECUTION & STREAMING
# ==============================================================================

def run_debate(motion: str, graph) -> dict:
    """Runs a single debate session and streams each speech as it is produced."""
    initial_state: DebateState = {
        "motion": motion,
        "speeches": [],
        "round": 0,
        "verdict": None,
        "winner": None,
    }

    print("\n" + "=" * 65, flush=True)
    print(f"🎙️  MOTION: {motion}", flush=True)
    print(f"⚙️  Rounds: {ROUNDS} | Model: {MODEL_NAME} | Temp: {TEMPERATURE}", flush=True)
    print("=" * 65 + "\n", flush=True)

    current_round = 1
    final_state = initial_state

    # Stream each node execution in LangGraph
    for event in graph.stream(initial_state, stream_mode="updates"):
        if "pro" in event:
            speech = event["pro"]["speeches"][-1]
            content = speech[len("PRO: "):] if speech.startswith("PRO: ") else speech
            print(f"🟢 PRO (Round {current_round}):", flush=True)
            print(f"   {content}\n", flush=True)

        elif "con" in event:
            speech = event["con"]["speeches"][-1]
            content = speech[len("CON: "):] if speech.startswith("CON: ") else speech
            print(f"🔴 CON (Round {current_round}):", flush=True)
            print(f"   {content}\n", flush=True)
            current_round += 1

        elif "judge" in event:
            verdict = event["judge"]["verdict"]
            winner = event["judge"]["winner"]
            winner_badge = "🟢 PRO" if winner == "PRO" else ("🔴 CON" if winner == "CON" else "🤝 TIE")
            print("-" * 65, flush=True)
            print("⚖️  JUDGE:", flush=True)
            for line in verdict.splitlines():
                if line.strip():
                    print(f"   {line.strip()}", flush=True)
            print(f"\n🏆 Winner: {winner_badge}", flush=True)
            print("=" * 65 + "\n", flush=True)
            final_state = event["judge"]

    return final_state


def main():
    """Interactive CLI loop."""
    print("=" * 65, flush=True)
    print("       🗣️  DEBATE AGENT — MULTI-AGENT CLI (LangGraph)", flush=True)
    print("=" * 65, flush=True)
    print("Enter a debate motion to begin.", flush=True)
    print("Type 'exit' or 'quit' to exit.\n", flush=True)

    try:
        graph = build_debate_graph()
    except Exception as e:
        print(f"❌ Error initializing LangGraph / Ollama: {e}", flush=True)
        print("Please ensure Ollama is running (`ollama serve`) and `llama3.2` is pulled.", flush=True)
        sys.exit(1)

    # Check if motion was passed as command line argument
    if len(sys.argv) > 1:
        motion_arg = " ".join(sys.argv[1:]).strip()
        if motion_arg:
            try:
                run_debate(motion_arg, graph)
            except Exception as e:
                print(f"❌ Error during debate: {e}", flush=True)
            return

    # Interactive Loop
    while True:
        try:
            motion = input("Motion: ").strip()
            if not motion:
                continue
            if motion.lower() in ("exit", "quit", "q"):
                print("Exiting debate arena. Goodbye!", flush=True)
                break

            run_debate(motion, graph)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye!", flush=True)
            break
        except Exception as e:
            print(f"\n❌ Error during debate: {e}\n", flush=True)


if __name__ == "__main__":
    main()
