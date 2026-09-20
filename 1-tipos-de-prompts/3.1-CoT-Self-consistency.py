from collections import Counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console

from utils import print_llm_result

load_dotenv()
console = Console()

msg1 = """
Question: Alice has 3 brothers and 2 sisters.
How many sisters does Alice's brother have?

Think step by step.
At the end, give only the final number after "Answer:".
"""

N = 5  # number of independent reasoning paths

# temperature > 0 so each sample can follow a different reasoning path
# (unlike 3-CoT.py, where temperature=0 keeps runs comparable)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
# llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# N independent calls: each path is sampled without seeing the others
responses = llm.batch([msg1] * N)

answers = []
for response in responses:
    print_llm_result(msg1, response)
    content = response.content
    if "Answer:" in content:
        answers.append(content.rsplit("Answer:", 1)[-1].strip().rstrip("."))

# Majority vote over the final answers, done in code instead of asking the model
votes = Counter(answers)
console.print(f"[bold]Votes:[/bold] {dict(votes)}")
if votes:
    final, count = votes.most_common(1)[0]
    console.print(f"[bold]Final answer:[/bold] {final} ({count}/{N} paths)")
else:
    console.print("[bold red]No path returned an 'Answer:' line[/bold red]")
