from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import print_llm_result

load_dotenv()
msg1 = """
Classify the log severity.

Input: "Disk usage at 85%."
Answer only with INFO, WARNING, or ERROR.
"""

msg2 = """
Classify the log severity.

Input: "Disk usage at 85%."
Think step by step about why this is INFO, WARNING, or ERROR. 
At the end, give only the final answer after "Answer:".
"""

msg3 = """
Question: How many "r" are in the word "strawberry"?
Answer only with the number of "r".
"""

msg4 = """
Question: How many "r" are in the word "strawberry"?
Explain step by step by breaking down each letter in bullet points, pointing out the "r" before giving the final answer. 
Give the final result after "Answer:".
"""

# temperature=0 keeps runs comparable, so differences come from the prompt (CoT or not)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatOpenAI(model="gpt-5-nano") # reasoning model: does not accept a custom temperature

prompts = [msg1, msg2, msg3, msg4]

# The calls are independent, so batch runs them in parallel
for prompt, response in zip(prompts, llm.batch(prompts)):
    print_llm_result(prompt, response)