from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader(r'8_Document_Loader\documents\data.txt')

docs = loader.load()

# print(docs[0].page_content)
# print(docs[0].metadata)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="explain me the below poem in short. poem - {poem}",
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'poem' : docs[0].page_content})
print(result)
