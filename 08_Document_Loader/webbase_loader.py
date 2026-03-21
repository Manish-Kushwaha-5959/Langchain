from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Enter you url here
url = r'https://n8n.io/?ps_partner_key=NGY0MTI3ZmE3NzMy&sid1=3060020017&ps_xid=utJUCpAmPUV8BC&gsxid=utJUCpAmPUV8BC&gspk=NGY0MTI3ZmE3NzMy&gad_source=1'

loader = WebBaseLoader(url)

docs = loader.load()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='answer the question based on the given text.\n{question} \n{text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

chain = prompt | model | parser

question = 'give me 5-6 important points from the given text'

result = chain.invoke({'question' : question, 'text' : docs[0].page_content})
print(result)