from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="Your are helpful chat customer service chat assistant")
]

while True:
    user_input = input("User : ")
    if (user_input == "exit"):
        break
    messages.append(HumanMessage(content=user_input))
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print(f"AI : {result.content}")

print(messages)