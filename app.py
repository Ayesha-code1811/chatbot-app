import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Base model from Hugging Face
model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Chat function
def generate_response(user_input):
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.replace(prompt, "").strip()
    return response

# Streamlit UI
st.set_page_config(page_title="📜 Legal Chatbot Assistant", layout="centered")
st.title("🧑‍⚖️ Law Chatbot")
st.markdown("Ask me anything about FIRs, criminal law, or general legal procedure.")

# User input
user_query = st.text_input("📝 Enter your legal question here:")

if user_query:
    with st.spinner("Thinking like a lawyer..."):
        answer = generate_response(user_query)
    st.success("🤖 " + answer)

