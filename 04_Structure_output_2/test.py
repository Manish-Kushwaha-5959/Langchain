from dotenv.variables import Variable
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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

prompt1 = template1.invoke({'topic' : 'DropShipping'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result.content})

result1 = model.invoke(prompt2)

print(result1.content)
