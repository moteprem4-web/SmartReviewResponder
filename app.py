import streamlit as st
import pickle
from langchain_community.llms import Ollama  # only if your workflow uses Ollama

st.set_page_config(page_title="AI Review Analyzer", page_icon="🧠")

st.title("🧠 AI Review Analyzer")
st.write("Enter a review and the AI will generate a response based on your workflow.")

# ----------------------------
# Load workflow from output.pk
# ----------------------------
@st.cache_resource
def load_workflow():
    try:
        with open("output.pkl", "rb") as f:
            workflow = pickle.load(f)
        return workflow
    except FileNotFoundError:
        st.error("output.pk not found. Make sure it exists in the project folder.")
        return None

workflow = load_workflow()

# ----------------------------
# User input
# ----------------------------
review = st.text_area("Enter your review here:")

# ----------------------------
# Button action
# ----------------------------
if st.button("Analyze Review"):
    if not workflow:
        st.error("Workflow not loaded. Cannot process review.")
    elif not review.strip():
        st.warning("Please enter a review first!")
    else:
        try:
            # Adjust this key based on your workflow state
            initial_state = {"review": review}  
            result = workflow.invoke(initial_state)
            
            st.subheader("AI Response")
            st.write(result.get("response", "No response generated"))

            # Optional: save last result
            with open("last_result.pkl", "wb") as f:

                pickle.dump(result, f)

        except Exception as e:
            st.error(f"Error running workflow: {e}")
