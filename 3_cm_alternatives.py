from langchain.chat_models import anthropic
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import creds

messages = [
    SystemMessage(content="Solve the answer."),
    HumanMessage(content="What is the capital of India?")
]

model = AzureChatOpenAI(api_key="")
result = model.invoke(messages)
print("Result from azure: ",result.content)


model2 = ChatOpenAI(api_key="")
result = model2.invoke(messages)
print("Result from GPT: ",result.content)  # this content gives only result not the tokens created