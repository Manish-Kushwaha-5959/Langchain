from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='generate 5 short questions answers from the following text \n {text}'
)

prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into a single document and formate it properly\n notes -> {notes}, quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
LangChain is an open-source orchestration framework designed to simplify the creation of applications powered by Large Language Models (LLMs). While models like GPT-4 or Gemini are powerful on their own, they are often limited by "knowledge cutoffs" and an inability to interact with external data or software. LangChain bridge this gap by providing a standardized "plumbing" system that allows developers to "chain" together different components. These components include Prompt Templates for consistent formatting, LLM Wrappers for switching between different AI providers, and Output Parsers to turn raw text into structured data like JSON or tables.

One of the framework's most critical features is its ability to handle Retrieval Augmented Generation (RAG). This process enables an AI to query specific, private data sources—such as a company’s internal PDFs or a SQL database—before generating a response, ensuring the output is grounded in factual, up-to-date information. Beyond simple data retrieval, LangChain introduces Agents, which are autonomous loops where the LLM can "decide" to use external tools, such as performing a web search or executing Python code, to solve complex multi-step problems. By managing state and memory, LangChain transforms a static chatbot into a dynamic assistant capable of maintaining long-term context and executing sophisticated workflows.
"""

result = chain.invoke({'text' : text})

print(result)

