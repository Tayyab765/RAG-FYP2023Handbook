import os
import streamlit as st
from ask import ask, retrieve_chunks
from datetime import datetime

# Where to save logs (same folder as this app.py)
LOG_PATH = os.path.join(os.path.dirname(__file__), "log.txt")

def log_turn(question: str, answer: str, sources: list[str]):
    """Append a Q/A turn to log.txt with timestamp and sources."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] Q: {question}\n")
            f.write("A:\n")
            f.write((answer or "").strip() + "\n")
            if sources:
                f.write("Sources: " + " | ".join(sources) + "\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        # Non-fatal: surface a warning but don't break the app
        st.warning(f"Failed to write to log.txt: {e}")

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="FAST-NUCES FYP Handbook Assistant",
    page_icon="📘",
    layout="centered"
)

st.title("📘 FAST-NUCES FYP Handbook RAG Assistant")
st.caption("Answers are grounded strictly in the 2023 FYP Handbook. Citations show page numbers.")


# ---------------------------------------------------------
# SIDEBAR: STATUS & HELP
# ---------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app retrieves the most relevant handbook passages and asks an LLM to answer using only that context."
    )
    key_present = bool(os.getenv("OPENAI_API_KEY"))
    st.markdown(
        f"OpenAI API key: {'✅ Detected' if key_present else '⚠️ Missing'}"
    )
    st.caption("Set `OPENAI_API_KEY` in your environment before running the app.")
    st.markdown("---")
    st.subheader("Tips")
    st.write("• Ask specific questions to get precise citations.")
    st.write("• Use the expander to inspect retrieved context chunks.")


# ---------------------------------------------------------
# SESSION STATE (chat-like history)
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {q, a, sources, chunks}


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
question = st.text_area(
    "Enter your question about the FYP Handbook:",
    placeholder="e.g., How are the FYP grading criteria weighted?",
    height=100,
)

col_ask, col_clear = st.columns([1, 1])
ask_clicked = col_ask.button("Ask", type="primary")
clear_clicked = col_clear.button("Clear", help="Clear the current conversation")

if clear_clicked:
    st.session_state.history = []
    st.rerun()

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant information and generating answer..."):
            answer, sources = ask(question)
            # Retrieve chunks again to show detailed context (page, score, text)
            try:
                chunks = retrieve_chunks(question)
            except Exception:
                chunks = []

        st.session_state.history.append(
            {"q": question.strip(), "a": answer, "sources": sources, "chunks": chunks}
        )
        # Log the interaction to log.txt
        log_turn(question.strip(), answer, sources)


# ---------------------------------------------------------
# RENDER CONVERSATION (latest first)
# ---------------------------------------------------------
if st.session_state.history:
    for turn_idx, turn in enumerate(reversed(st.session_state.history), start=1):
        st.markdown("---")
        st.markdown(f"**Q{turn_idx}:** {turn['q']}")

        st.subheader("Answer")
        st.write(turn["a"])  # may include model error text handled upstream

        st.subheader("Sources (Page References)")
        if turn["sources"]:
            for s in turn["sources"]:
                st.write(f"• {s}")
        else:
            st.write("No sources found (similarity below threshold).")

        with st.expander("Show Retrieved Context Chunks"):
            chunks = turn.get("chunks", [])
            if not chunks:
                st.write("No context available.")
            else:
                for c in chunks:
                    page = c.get("page", "?")
                    score = c.get("similarity", 0.0)
                    st.markdown(f"**p.{page} — score={score:.3f}**")
                    st.write(c.get("text", ""))
                    st.markdown("---")
else:
    st.info("Ask a question above to get started.")
