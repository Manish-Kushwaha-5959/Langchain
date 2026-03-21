from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text_python = """
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
"""

text_markdown = """
# Needle In A Haystack - Pressure Testing LLMs

Supported model providers: OpenAI, Anthropic

A simple 'needle in a haystack' analysis to test in-context retrieval ability of long context LLMs.

Get the behind the scenes on the [overview video](https://youtu.be/KwRRuiCCdmc).

![GPT-4-128 Context Testing](img/NeedleHaystackCodeSnippet.png)

git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git

## The Test
1. Place a random fact or statement (the 'needle') in the middle of a long context window (the 'haystack')
2. Ask the model to retrieve this statement
3. Iterate over various document depths (where the needle is placed) and context lengths to measure performance

"""

splitter_python = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 200,
    chunk_overlap=0
)

splitter_markdown = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap = 0
)

result_python = splitter_python.split_text(text_python)
result_markdown = splitter_markdown.split_text(text_markdown)

print("PYTHON")
print(len(result_python))
print(result_python[0])
print(result_python[1])

print()
print()

print("MARKDOWN")
print(len(result_markdown))
print(result_markdown[0])
print(result_markdown[1])