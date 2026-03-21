from torchgen.api.cpp import name
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description="Fact 1 about topic"),
    ResponseSchema(name="fact2", description="Fact 2 about topic"),
    ResponseSchema(name="fact3", description="Fact 3 about topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="give 3 facts about the {topic}.\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke({'topic' : 'xyz'})
# print(prompt)

# chain = template | model | parser

# result = chain.invoke({'topic' : 'black hole'})

# print(result)

prompt = template.invoke({'topic' : 'black hole'})
result = model.invoke(prompt)
print(result.content)
