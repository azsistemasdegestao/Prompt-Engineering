from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import print_llm_result

load_dotenv()

msg1 = """
You are a senior software engineer.
A user reports that an API request to the endpoint `/users` is taking 5 seconds to respond, which is too slow.
Think in a Tree of Thought manner:
- Generate at least 3 different possible causes for this latency.
- For each cause, reason step by step about how likely it is and how you would verify it.
- Then compare the branches and choose the most plausible one as the primary hypothesis.
- Finish with a recommended next action to debug or fix the issue.
"""

msg2 = """
You are designing a service that processes millions of images daily.
Think in a Tree of Thought manner:
- Generate at least 3 different architecture options.
- For each option, reason step by step about scalability, cost, and complexity.
- Compare the options.
- Choose the best trade-off and explain why it is superior to the others.
- Finish with "Final Answer: " + the chosen option.
"""

# Same scenario as msg2, but asking for the leaf instead of the whole tree.
# A non-reasoning model writes its reasoning in the output, so suppressing the
# output also suppresses the tree: this prompt only does ToT on a reasoning
# model, where the branches live in hidden tokens.
msg3 = """
You are designing a service that processes millions of images daily.
Think in a Tree of Thought manner:
- Think about at least 3 different architecture options.
- For each option, reason step by step about scalability, cost, and complexity.
- Compare the options.
- Choose the best trade-off.

- OUTPUT ONLY THE CHOSEN OPTION, IN 6 WORDS OR LESS, WITHOUT ANY OTHER TEXT.
"""

# temperature=0 keeps runs comparable, so differences come from the prompt
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# reasoning model: does not accept a custom temperature
llm_reasoning = ChatOpenAI(model="gpt-5-nano")

prompts = [msg1, msg2]

# The calls are independent, so batch runs them in parallel
for prompt, response in zip(prompts, llm.batch(prompts)):
    print_llm_result(prompt, response)

# msg3 goes to the reasoning model, the only one that can hide the tree
print_llm_result(msg3, llm_reasoning.invoke(msg3))
