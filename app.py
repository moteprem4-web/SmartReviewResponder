import streamlit as st
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph, START, END
import os
import json

# ==============================
# 1. Configuration
# ==============================
load_dotenv()
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Review Response Bot", page_icon="💬")

HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def get_llm():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.01,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
    )
    return ChatHuggingFace(llm=llm_endpoint)

llm = get_llm()

# ==============================
# 2. Schemas
# ==============================
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"]

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

# ==============================
# 3. State
# ==============================
class ReviewState(TypedDict):
    review: str
    sentiment: str
    diagnosis: dict
    response: str

# ==============================
# 4. Helpers
# ==============================
def clean_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    if "{" in text:
        text = text[text.find("{"):text.rfind("}") + 1]
    return json.loads(text)

# ==============================
# 5. Nodes
# ==============================
def find_sentiment(state: ReviewState):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a sentiment analyzer.\n"
        "Return ONLY valid JSON like this:\n"
        '{ "sentiment": "positive" }\n'
        "OR\n"
        '{ "sentiment": "negative" }\n'
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{state['review']}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    response = llm.invoke(prompt)
    data = clean_json(response.content)
    parsed = SentimentSchema(**data)
    return {"sentiment": parsed.sentiment}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    return "positive_response" if state["sentiment"] == "positive" else "run_diagnosis"

def positive_response(state: ReviewState):
    prompt = f"Write a warm, friendly thank-you response for this review:\n{state['review']}"
    response = llm.invoke(prompt)
    return {"response": response.content}

def run_diagnosis(state: ReviewState):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a diagnostic assistant.\n"
        "Choose EXACTLY ONE value for each field.\n\n"
        "Return ONLY valid JSON like this example:\n"
        '{\n'
        '  "issue_type": "Bug",\n'
        '  "tone": "angry",\n'
        '  "urgency": "high"\n'
        '}\n\n'
        "Valid values:\n"
        "- issue_type: UX, Performance, Bug, Support, Other\n"
        "- tone: angry, frustrated, disappointed, calm\n"
        "- urgency: low, medium, high\n\n"
        "Do NOT combine values. Do NOT use | symbols.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{state['review']}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    response = llm.invoke(prompt)
    data = clean_json(response.content)
    parsed = DiagnosisSchema(**data)
    return {"diagnosis": parsed.model_dump()}

def negative_response(state: ReviewState):
    d = state["diagnosis"]
    prompt = (
        f"You are a customer support agent.\n"
        f"Issue: {d['issue_type']}\n"
        f"Tone: {d['tone']}\n"
        f"Urgency: {d['urgency']}\n\n"
        f"Write an empathetic, helpful response to this review:\n"
        f"{state['review']}"
    )
    response = llm.invoke(prompt)
    return {"response": response.content}

# ==============================
# 6. Workflow
# ==============================
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

# ==============================
# 7. Streamlit UI
# ==============================
st.title("💬 Review Response Generator")

if "history" not in st.session_state:
    st.session_state.history = []

review = st.text_area("Enter customer review:")

if st.button("Generate Response", type="primary"):
    if review.strip():
        with st.spinner("Analyzing review..."):
            try:
                result = workflow.invoke({"review": review})
                st.session_state.history.append(
                    {"review": review, "response": result["response"]}
                )
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a review.")

for item in reversed(st.session_state.history):
    st.markdown(f"**Customer:** {item['review']}")
    st.success(f"**Agent:** {item['response']}")
    st.divider()
