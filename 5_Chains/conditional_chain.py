from narwhals.dtypes import TemporalType
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
str_parser = StrOutputParser()

class Review(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="sentiment of the text.")

pyd_parser = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template="Classifie the sentiment of the feedback in positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction' : pyd_parser.get_format_instructions()}
)

classifier = prompt1 | model | pyd_parser

prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier | branch_chain

result = chain.invoke({'feedback' : 'this smartphone is very good'})

print(result)