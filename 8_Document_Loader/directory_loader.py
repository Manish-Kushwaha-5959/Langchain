from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path=r'8_Document_Loader\documents',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(len(docs))

# for document in docs:
#     print(document.metadata)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

