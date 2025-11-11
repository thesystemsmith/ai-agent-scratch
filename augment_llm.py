import os
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='qwen2.5:3b-instruct',
    temperature = 0
)

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(None, description="Why this query is relevant.")


structured_llm = llm.with_structured_output(SearchQuery)

output = structured_llm.invoke('How does Calcium CT score relate to high cholesterol?')
print("\n[Structured Output]")
print(output)                 # -> a Pydantic object (SearchQuery)
print(output.model_dump())    # -> dict

# tooling
from langchain_core.tools import tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a*b

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke('what is 2 times 3?')
print("\n[LLM Message]")
print(msg)

print("\n[Tool Calls]")
print(msg.tool_calls)

for call in msg.tool_calls or []:
    if call["name"] == "multiply":
        # call["args"] is already a dict with validated args
        result = multiply.invoke(call["args"])
        print("\n[Tool Result]")
        print(result)