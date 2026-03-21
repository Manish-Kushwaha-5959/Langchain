from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are professional {domain} expert'),
    ('human', 'Explain about the {topic} in simple language')
])

prompt = chat_template.invoke({
    'domain' : 'Computer Science',
    'topic' : 'DSA'
})

print(prompt)