from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

template = ChatPromptTemplate([
    ('system', 'you are a helpful customer service chat assisstant'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{query}')
])

chat_history = []
with open("prompt/chat_history.txt") as f:
    chat_history.extend(f.readlines())

# print(chat_history)

query = "where is my order"
prompt = template.invoke({
    "chat_history" : chat_history,
    "query" : query
})

print(prompt)