import requests

from langchain import OpenAI
from langchain.agents import initialize_agent, load_tools, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun


OPENAI_API_KEY = "sk-..."

llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.8,
    model_name="text-davinci-003"
)

prompt = PromptTemplate(
  input_variables=["query"],
  template="You are New Native Internal Bot. Help users with their important tasks, like a professor in a particular field. Query: {query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# DuckDuckGo for web search START
search = DuckDuckGoSearchRun()

# Web Search Tool
search_tool = Tool(
    name = "Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions."
) 
# DuckDuckGo for web search END


# wolfram alpha for math calculations START
class WA:
  """
    Wolfram|Alpha API
  """
  def __init__(self, app_id):
    self.url = f"http://api.wolframalpha.com/v1/result?appid={app_id}&i="

  def run(self, query):
    query = query.replace("+", " plus ").replace("-", " minus ") # '+' and '-' are used in URI and cannot be used in request
    result = requests.post(f"{self.url}{query}")

    if not result.ok:
      raise Exception("Cannot call WA API.")

    return result.text



WA_API_KEY = "..." # You can get it here: https://products.wolframalpha.com/api/

wa = WA(app_id=WA_API_KEY)

wa_tool = Tool(
    name="Wolfram|Alpha API",
    func=wa.run,
    description="Wolfram|Alpha API. It's super powerful Math tool. Use it for simple & complex math tasks."
)
# wolfram alpha for math calculations END

# agent START
agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=[wa_tool, search_tool],
    llm=llm,
    verbose=True, # I will use verbose=True to check process of choosing tool by Agent
    max_iterations=3
)


# agent END



# run test
r_1 = agent("What is spasex?")
print(f"Final answer: {r_1['output']}")

r_2 = agent("Integral of x * (log(x)^2)")
print(f"Final answer: {r_2['output']}")
