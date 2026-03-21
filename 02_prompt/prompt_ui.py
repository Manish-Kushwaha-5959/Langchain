import streamlit as st
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
#import os

#os.environ['HF_HOME'] = 'D:/huggingface_cache'

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("Paper Summerizer")

paper_input = st.selectbox("Select Researh Paper", ["Attenion is all you need", "BERT: Pre-training of deep bidirectional Transformers", "GPT-3: Language model are few shot learners", "Diffusion model beats GANs on image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner Freindly", "Technical", "code oriented", "mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragrphs)", "Long (detailed Explanation)"])

#template
template = load_prompt("prompt/template.json")

if st.button("Submit"):
    chain = template | model
    result = chain.invoke({
        'paper_input' : paper_input,
        'style_input' : style_input,
        'length_input' : length_input
    })
    st.write(result.content)