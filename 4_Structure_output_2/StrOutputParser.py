from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# prompt 1

template1 = PromptTemplate(
    template='Write a Detailed report on the {topic}.',
    input_variables=['topic']
)

# prompt 2

template2 = PromptTemplate(
    template='Write a 5 line summary of the below text : \n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'AI Agents'})

print(result)