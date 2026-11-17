
import streamlit as st
from src.chain import answer_question

st.set_page_config(
    page_title="Indian Tax RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("🇮🇳 Indian Tax RAG Assistant")
st.write(
    "Ask questions about Indian Income Tax and GST. "
    "Answers are generated from the indexed sources."
)

st.info("General information, not tax advice.")

# Select the source category
act_options = {
    "All indexed sources": None,
    "Income-tax Act, 2025": "IT-2025",
    "GST": "GST",
    "SVLDRS, 2019": "SVLDRS-2019"
}

selected_act = st.selectbox(
    "Choose a source",
    list(act_options.keys())
)

# Question input
with st.form("question_form"):
    question = st.text_area(
        "Enter your tax question:",
        placeholder="Example: When did the Income-tax Act, 2025 come into force?"
    )

    submitted = st.form_submit_button("Get Answer")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching sources and generating an answer..."):
                result = answer_question(
                    question,
                    k=6,
                    act=act_options[selected_act]
                )

            # Support tuple or dictionary results
            if isinstance(result, tuple):
                answer, sources = result
            elif isinstance(result, dict):
                answer = result.get("answer", "")
                sources = result.get("sources", [])
            else:
                answer = str(result)
                sources = []

            st.subheader("Answer")
            st.markdown(answer)

            st.subheader("Sources")

            if sources:
                for source in sources:
                    with st.expander(
                        f"{source.get('citation', '[Source]')} "
                        f"{source.get('title', 'Untitled source')}"
                    ):
                        st.write("Act:", source.get("act", "Not available"))
                        st.write("Section:", source.get("section", "Not available"))
                        st.write("Page:", source.get("page", "Not available"))
                        st.write("As of:", source.get("as_of", "Not available"))

                        url = source.get("url")
                        if url:
                            st.markdown(f"[Open source]({url})")
            else:
                st.write("No source details were returned.")

            st.caption("General information, not tax advice.")

        except Exception as error:
            st.error(f"Something went wrong: {error}")
