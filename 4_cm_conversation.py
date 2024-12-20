# this program basically builds a coversation with chat models and stores the chat history.
# so instead of using gpt we access them here through the aichat models.

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import creds

model = ChatOpenAI(api_key= creds.api_key)

chat_history =[]
system_message = SystemMessage(content="Help my user and answer them.")
chat_history.append(system_message)

while True:        # chat loop
    user_ques = input("Your questions: ")    
    if user_ques.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=user_ques))

    result = model.invoke(chat_history)       # here we use chat_history becoz to have a connected conversation 
    response = result.content
    chat_history.append(AIMessage(content=response))

    print("AI Result: ", response)

print("______________CHAT HISTORY______________")
print(chat_history)