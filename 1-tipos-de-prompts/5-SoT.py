import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.text import Text

from utils import print_llm_result

load_dotenv()
console = Console()

# msg1 and msg2 put "Step 1" and "Step 2" inside a single prompt: the model
# decodes the skeleton and the expansion in one sequential stream, so this is a
# structured CoT, not SoT. Kept here as the contrast to the real technique below.
msg1 = """
You are a senior backend engineer. A junior developer asked you how to optimize SQL queries for better performance.
Follow the Skeleton of Thought approach:

Step 1: Generate only the skeleton of your answer in 3-5 concise bullet points.
Step 2: Expand each bullet point into a clear and detailed explanation with examples.
Make sure the final answer is structured and easy to follow.
"""

msg2 = """
You are a software architect. I want you to produce an Architecture Decision Record (ADR) about choosing PostgreSQL instead of MongoDB.

Follow the Skeleton of Thought approach:
Step 1: First, output only the skeleton of the ADR as section headers (no explanations yet).
Use the standard ADR structure with 5 sections: Context, Decision, Alternatives Considered, Consequences, References.
Step 2: After showing the skeleton, expand each section with clear and detailed content.
Keep the final ADR professional, structured, and easy to read.
"""

# Real Skeleton of Thought: two stages of calls instead of two steps in one
# prompt. Stage 1 produces the skeleton, stage 2 expands each point in its own
# call, and the N expansions are decoded in parallel. That parallel decoding is
# where the latency gain of the technique comes from.
skeleton_prompt = """
You are a senior Go developer planning a REST API for managing products in Go.
The API must implement CRUD operations for products with fields: id, name, description, price, stock.

Output ONLY the skeleton of the solution: 6-8 bullet points, one per line, each
starting with "- " and no longer than 10 words.
Cover: data model as Go structs, HTTP framework or net/http, routing, handlers,
validation, database layer, error handling, and project structure.
Do not explain or expand anything yet.
"""

# The full skeleton goes into every expansion so each call knows what the other
# points cover and does not repeat them.
expansion_prompt = """
You are a senior Go developer planning a REST API for managing products in Go
(CRUD over id, name, description, price, stock).

This is the full outline of the answer:
{skeleton}

Expand ONLY this point: {point}

Write 3-5 sentences plus a short idiomatic Go snippet when it helps.
Do not repeat the other points and do not write a heading.
"""

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)")


def parse_skeleton(text):
    """Return the bullet points of a skeleton response, one string per point."""
    points = []
    for line in text.splitlines():
        match = BULLET.match(line)
        if match:
            points.append(match.group(1).strip())
    return points


# temperature=0 keeps runs comparable, so differences come from the prompt
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatOpenAI(model="gpt-5-nano") # reasoning model: does not accept a custom temperature

prompts = [msg1, msg2]

# The calls are independent, so batch runs them in parallel
for prompt, response in zip(prompts, llm.batch(prompts)):
    print_llm_result(prompt, response)

# Stage 1: a single call, and the answer is only the outline
skeleton_response = llm.invoke(skeleton_prompt)
print_llm_result(skeleton_prompt, skeleton_response)

skeleton = skeleton_response.content
points = parse_skeleton(skeleton)

if not points:
    console.print("[bold red]No bullet points found in the skeleton[/bold red]")
else:
    # Stage 2: one call per point, sent together so they decode in parallel
    expansions = llm.batch(
        [expansion_prompt.format(skeleton=skeleton, point=point) for point in points]
    )

    # Stitch the parts back into a single document, in skeleton order
    final = "\n\n".join(
        f"## {point}\n{expansion.content.strip()}"
        for point, expansion in zip(points, expansions)
    )

    console.print(f"[bold]Skeleton points:[/bold] {len(points)}")
    console.print(f"[bold]Expansion calls:[/bold] {len(expansions)} (in parallel)")
    console.print(Text("FINAL ANSWER:", style="bold green"))
    # Text() instead of markup: Go snippets contain [] that rich would parse as tags
    console.print(Text(final, style="bold blue"))
