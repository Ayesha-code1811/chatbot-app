import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use base model
model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Generate chatbot reply
def generate_response(user_input):
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")  # ✅ FIXED: removed .to(model.device)

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
st.set_page_config(page_title="📜 Legal Chatbot Assistant", layout="centered")
st.title("🧑‍⚖️ Law Chatbot")
st.markdown("Ask me anything about FIRs, criminal law, or general legal procedure.")

user_input = st.text_input("📝 Enter your legal question here:")

if user_input:
    with st.spinner("Analyzing your question..."):
        answer = generate_response(user_input)
        st.success("🤖 " + answer)
