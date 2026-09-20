import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.text import Text

from utils import print_llm_result

load_dotenv()
console = Console()

# msg puts the decomposition and the solving inside a single prompt: the model
# decodes the to-do list and every solution in one stream, so this is a
# structured CoT, not Least-to-Most. The checkboxes are cosmetic — a model
# cannot update text it already emitted, it can only reprint the whole list.
# Kept here as the contrast to the real technique below.
msg = """
You are a senior Go backend engineer.

Problem: We need to design a URL shortener service in Go.

Use the Least-to-Most Prompting method:
1. Start by listing the subproblems that need to be solved as a to-do list. Use markdown checkboxes: [ ] for pending, [x] for completed.
2. As you solve each subproblem, update the to-do list by marking it as [x] and write the solution right below it.
3. Continue until all items are solved.
4. At the end, combine all the solutions into a final integrated design for the URL shortener.

Constraints:
- Service must be implemented in Go.
- Short URLs must be unique and easy to generate.
- Must support endpoints: shorten a URL, retrieve the original URL.
- Use an in-memory store at first, but mention how it could scale with a database.
- Include minimal validation and error handling.
- Keep explanations concise and structured.

Output format:
- To-do list with checkboxes (updating as you progress)
- Each solved subproblem explained with reasoning and minimal Go code snippets
- Final combined design
"""

CONSTRAINTS = """- Service must be implemented in Go.
- Short URLs must be unique and easy to generate.
- Must support endpoints: shorten a URL, retrieve the original URL.
- Use an in-memory store at first, but mention how it could scale with a database.
- Include minimal validation and error handling."""

# Real Least-to-Most: stage 1 decomposes the problem, stage 2 solves one
# subproblem per call. Unlike 5-SoT.py these calls cannot be batched: each one
# receives the subproblems already solved, so every answer builds on the ones
# before it. That accumulation is the technique — and the reason the ordering
# asked for below (least dependent first) is part of the prompt.
decompose_prompt = f"""
You are a senior Go backend engineer.

Problem: design a URL shortener service in Go.

Constraints:
{CONSTRAINTS}

List ONLY the subproblems that must be solved, ordered from the one that depends
on nothing to the one that depends on all the others. 4-6 items, one per line,
each starting with "- " and no longer than 10 words.
Do not solve anything yet.
"""

solve_prompt = """
You are a senior Go backend engineer designing a URL shortener in Go.

Constraints:
{constraints}

Full list of subproblems:
{subproblems}

Solutions already produced, in order:
{solved}

Solve ONLY this subproblem: {point}

Build on the solutions above instead of restating them, and stay consistent with
the types and names they introduced. Write 3-5 sentences plus a minimal
idiomatic Go snippet. Do not write a heading.
"""

integrate_prompt = """
You are a senior Go backend engineer.

These are the subproblems of a URL shortener in Go, solved one at a time:

{solved}

Combine them into a single coherent design: a short overview, the final package
layout, and the endpoints. Keep it concise and do not repeat the snippets in
full.
"""

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)")


def parse_subproblems(text):
    """Return the bullet points of a decomposition, one string per subproblem."""
    points = []
    for line in text.splitlines():
        match = BULLET.match(line)
        if match:
            points.append(match.group(1).strip())
    return points


def run(prompt, llm):
    """Call the model, print prompt, response and tokens, and return the text."""
    response = llm.invoke(prompt)
    print_llm_result(prompt, response)
    return response.content.strip()


# reasoning model: does not accept a custom temperature
llm = ChatOpenAI(model="gpt-5-mini")

# The single-prompt version first, as the baseline to compare against
run(msg, llm)

# Stage 1: decompose only, no solving
subproblems_text = run(decompose_prompt, llm)
subproblems = parse_subproblems(subproblems_text)

if not subproblems:
    console.print("[bold red]No subproblems found in the decomposition[/bold red]")
else:
    # Stage 2: one call per subproblem, in sequence. The prompt grows every
    # iteration because it carries the previous answers — that growing input is
    # what Least-to-Most costs, and what makes the last answers the best ones.
    solved_parts = []
    for i, point in enumerate(subproblems, start=1):
        solved = "\n\n".join(solved_parts) or "(nothing yet, this is the first subproblem)"
        solved_parts.append(
            f"### {point}\n"
            + run(
                solve_prompt.format(
                    constraints=CONSTRAINTS,
                    subproblems=subproblems_text,
                    solved=solved,
                    point=point,
                ),
                llm,
            )
        )
        # The checklist lives in Python, which can actually update it
        boxes = " ".join("[x]" if j <= i else "[ ]" for j in range(1, len(subproblems) + 1))
        console.print(Text(f"Progress: {i}/{len(subproblems)}  {boxes}", style="bold yellow"))

    # Stage 3: fold the accumulated solutions into one design
    final = run(integrate_prompt.format(solved="\n\n".join(solved_parts)), llm)

    console.print(f"[bold]Subproblems:[/bold] {len(subproblems)} (solved in sequence)")
    console.print(Text("FINAL DESIGN:", style="bold green"))
    # Text() instead of markup: Go snippets contain [] that rich would parse as tags
    console.print(Text(final, style="bold blue"))
