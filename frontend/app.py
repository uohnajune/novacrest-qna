import streamlit as st
import requests

BACKEND_URL = "http://localhost:7860"

st.set_page_config(page_title="NovaCrest QnA", page_icon="💼", layout="wide")

with st.sidebar:
    st.header("⚙️ Parameters")
    k = st.slider("k (chunks to retrieve)", 1, 10, 2)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, step=64)

st.title("💼 NovaCrest QnA")
st.write("Ask a question about the Salesforce manuals.")

response_type = st.radio("Response mode", ["Answer only", "Answer + Relevant Chunks"], horizontal=True)
query = st.text_input("Your question",
                      placeholder="e.g., Can I use Salesforce on mobile devices, and are there any editions where mobile is not available?")

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = "/v1/answer" if response_type == "Answer only" else "/v1/answer_with_relevant_chunks"
        payload = {"query": query, "k": k, "temperature": temperature, "max_tokens": max_tokens}
        try:
            with st.spinner("Thinking..."):
                response = requests.post(BACKEND_URL + endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])
                for i, chunk in enumerate(result.get("relevant_chunks", []), start=1):
                    with st.expander(f"Chunk {i}"):
                        st.write(chunk)
            else:
                st.error(f"Unable to get a response from the RAG API. Status code: {response.status_code}")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the RAG API. Please make sure the backend service is running.")
