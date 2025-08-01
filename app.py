import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use base model from Hugging Face
model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Chat function
def generate_response(user_input):
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")  # No device handling
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.replace(prompt, "").strip()

# Streamlit UI
st.set_page_config(page_title="📜 Law Chatbot", layout="centered")
st.title("🧑‍⚖️ Legal Assistant Chatbot")
st.markdown("Ask me anything related to law, FIR, criminal or civil procedure.")

question = st.text_input("📝 Type your legal question below:")

if question:
    with st.spinner("Analyzing your question..."):
        answer = generate_response(question)
        st.success("🤖 " + answer)
