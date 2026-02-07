import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ------------------------
# OpenAI (MLX) client
# ------------------------
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy"
)

MODEL = "sov30b-feb6-dwq-2k"

# ------------------------
# Helpers
# ------------------------
def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def stream_chat(messages, max_tokens=8192):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )

    output = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            output += delta.content
            yield delta.content
    return output


# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Document Chat (MLX)", layout="wide")
st.title("📄 Chat with your document (MLX streaming)")

if "doc_text" not in st.session_state:
    st.session_state.doc_text = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# Upload
# ------------------------
uploaded = st.file_uploader(
    "Upload a document",
    type=["txt", "md", "pdf"]
)

if uploaded and st.session_state.doc_text is None:
    with st.spinner("Reading document..."):
        st.session_state.doc_text = read_file(uploaded)

    st.success("Document loaded")

    # ------------------------
    # Streaming summary
    # ------------------------
    st.subheader("🧠 Document summary")

    summary_box = st.empty()
    summary_text = ""

    summary_prompt = [
        {
            "role": "system",
            "content": "You summarize documents clearly and concisely."
        },
        {
            "role": "user",
            "content": f"Summarize the following document:\n\n{st.session_state.doc_text}"
        }
    ]

    for token in stream_chat(summary_prompt, max_tokens=512):
        summary_text += token
        summary_box.markdown(summary_text)

    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer questions using ONLY the document below.\n\n"
                f"{st.session_state.doc_text}"
            )
        }
    ]

# ------------------------
# Chat UI
# ------------------------
if st.session_state.doc_text:
    st.subheader("💬 Chat with the document")

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the document"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_box = st.empty()
            response_text = ""

            for token in stream_chat(st.session_state.messages, max_tokens=1024):
                response_text += token
                response_box.markdown(response_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

