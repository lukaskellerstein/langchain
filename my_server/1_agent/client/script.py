from langchain.prompts.chat import ChatPromptTemplate
from langserve import RemoteRunnable

# model
llm = RemoteRunnable("http://localhost:8000/agent-1/")

# -----------------------------------------------------------------

# Invoke
prompt = {"input": "I am in London, what should I wear?"}
result = llm.invoke(prompt)
print("---- Answer 1 ----")
print(type(result))
print(result)
