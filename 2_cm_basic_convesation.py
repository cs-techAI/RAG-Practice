from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import creds

model = ChatOpenAI(api_key= creds.api_key)

messages = [
    SystemMessage(content="Solve the problems"),
    HumanMessage(content = "What is 81 divided by 9?"),
    AIMessage(content= "81 divided by 9 is 9."),
    HumanMessage(content= "What is 10 times 5.")
]

result = model.invoke(messages)
print(result.content)