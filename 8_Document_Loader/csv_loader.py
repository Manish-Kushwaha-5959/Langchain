from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


loader = CSVLoader(r'8_Document_Loader\documents\data.csv')

docs = loader.load()

data = [document.page_content for document in docs]

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Analyse the below data and give insights from it, properly formated in points for better readability.\n{data}',
    input_variables=['data']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'data' : data})
print(result)