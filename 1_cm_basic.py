from langchain_openai import ChatOpenAI
import creds

model = ChatOpenAI(api_key=creds.api_key)

result = model.invoke("What is 81 divided by 9?")
print("Full result: \n", result)
print("Content Only: \n", result.content)