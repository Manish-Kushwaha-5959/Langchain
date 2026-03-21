from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(BaseModel):
    summary : str
    sentiment : str

structure_model = model.with_structured_output(Review)

result = structure_model.invoke("the hardware is great but the mobile performace is slow, rest all is great")

print(result)