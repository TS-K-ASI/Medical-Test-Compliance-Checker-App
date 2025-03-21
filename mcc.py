import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time

load_dotenv()

def medical_compliance_checker(region, user_input):
    prompt = f"""
    You are a compliance expert specializing in medical regulatory standards. Evaluate the following medical claim for compliance based on {region} regulations. Return either 'Compliant' or 'Non-Compliant' along with a brief reason only if non-compliant.

    Example Medical Claim: This drug guarantees 100% effectiveness in curing diabetes.

    Answer Format:

    Classification: [Compliant / Non-Compliant]

    Explanation: [Brief reason for classification only if Non-Compliant]

    Example Classification: Non-Compliant

    Example Explanation: Absolute claims are not allowed.

    Now, Evaluate the following medical claim:

    Medical Claim: {user_input}

    Classification: 

    Explanation: 
    """

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.95,
        top_k=40,
        max_output_tokens=4096,
        response_mime_type="text/plain",
    )

    model_response = ""
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # print(chunk.text, end="", flush=True)
        model_response += chunk.text

    classification = model_response.split("Classification:")[1].split("\n")[0].strip()
    explanation = model_response.split("Explanation:")[1].strip()

    return classification, explanation

# Streamlit UI
if "messages" not in st.session_state:
    st.session_state["messages"] = [] 
    
st.set_page_config(page_title="Medical Compliance Checker", layout="centered")
st.title("Medical Compliance Checker")

# Chatbot UI
st.subheader("Chat with Medical Compliance AI")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
region = st.selectbox("Select Regulatory Region:", ["FDA (US)", "EMA (Europe)", "HSA (Singapore)"])
user_input = st.chat_input("Enter a medical claim...")

if user_input:
    # Store user message in chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing..."):
        classification, explanation = medical_compliance_checker(region, user_input)

    # Store response in chat history
    response_text = f"**Classification:** {classification}"
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.write(response_text)

    if classification == "Non-Compliant":
        explanation_text = ""
        with st.chat_message("assistant"):
            explanation_container = st.empty()
            
            for char in explanation:
                explanation_text += char
                explanation_container.write(explanation_text)
                time.sleep(0.03)
        
        # Store explanation message in chat history
        st.session_state.messages.append({"role": "assistant", "content": explanation})