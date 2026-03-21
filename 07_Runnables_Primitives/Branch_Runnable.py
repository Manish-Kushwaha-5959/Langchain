from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='identify to which domain the topic - {topic} is most related to from fictional or scientific and if it is scientific provide a detailed report with more then 200 words and less than 300 words and if it is related to fiction give a persona related to this fictional character in less then 200 words',
    input_variables=['topic']
)


parser = StrOutputParser()

input_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>2000, RunnableLambda(lambda x: len(x.split()))),
    RunnablePassthrough()
)

final_chain = input_chain | branch_chain

result = final_chain.invoke("fictional superhero")
print(result)

