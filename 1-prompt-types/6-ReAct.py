from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.text import Text

from utils import print_llm_result

load_dotenv()
console = Console()

# msg1 and msg2 are SIMULATED ReAct: there is no tool, so the model writes the
# Action and then writes its own Observation. Both scenarios are closed worlds
# (everything needed is already in the prompt), so an Observation can legitimately
# restate the context, and the prompts forbid inventing anything beyond it.
# The real technique, where an Observation is the return value of a tool the code
# actually ran, is case 3 at the bottom of this file.
msg1 = """
You are a Go backend engineer helping debug a REST API.
Use the ReAct style reasoning: alternate between "Thought:" (your reasoning) and
"Action:" (a concrete step or check you would perform).
After each action, write "Observation:" with what that step reveals.
At the end, conclude with "Final Answer:" as your recommended fix.

You have no tools in this exercise. An Observation may only restate what the
context below already contains. If an action would need information that is not
in the context (error logs, database schema, database driver), write
"Observation: not available in the context" and do not guess what it would say.

Context: A user reports that the endpoint `POST /products` always returns HTTP 500.

Here is the handler code for `POST /products`:

```go
func CreateProduct(w http.ResponseWriter, r *http.Request) {
    var product Product
    err := json.NewDecoder(r.Body).Decode(&product)
    if err != nil {
        http.Error(w, "Bad Request", http.StatusBadRequest)
        return
    }

    stmt, err := db.Prepare("INSERT INTO products (id, name, description, price, stock) VALUES (?, ?, ?, ?, ?)")
    if err != nil {
        log.Fatal(err)
    }

    _, err = stmt.Exec(product.ID, product.Name, product.Description, product.Price, product.Stock)
    if err != nil {
        log.Println("Error during Exec:", err)
        http.Error(w, "Internal Server Error", http.StatusInternalServerError)
        return
    }

    w.WriteHeader(http.StatusCreated)
}

type Product struct {
    ID          string  `json:"id"`
    Name        string  `json:"name"`
    Description string  `json:"description"`
    Price       string  `json:"price"`
    Stock       int     `json:"stock"`
}
```
"""

msg2 = """
You are a travel planner helping a family choose the best way to go from Orlando to New York.
Use the ReAct style reasoning: alternate between "Thought:" (your reasoning) and
"Action:" (a step such as checking travel time, cost, or convenience).
After each action, write "Observation:" with what that step reveals.
At the end, conclude with "Final Answer:" as your recommendation.

You have no tools in this exercise. Every number you need is in the context below.
An Observation may only restate or compute from that context: do not invent prices,
schedules or availability.

Context:
- The family has 2 adults and 2 children (ages 5 and 8).
- Budget: max $1,000 for transport (not including hotel).
- They must arrive on July 10 in the evening.
- Options:
  - **Flight**: $220 per person round trip, 3-hour flight, plus $80 total in baggage fees.
  - **Train**: $150 per person round trip, 20-hour journey, with onboard WiFi and beds available for $50 extra per person.
  - **Car rental**: $60/day for 6 rental days, 2 days of driving each way, gas + tolls estimated $250 total. Kids get restless on long trips.

Other details:
- The kids' school finishes on July 9 at noon, so they cannot leave before then.
- Parents prefer not to arrive too tired, since they have a family wedding on July 11 in the morning.

Start your reasoning now.
"""

# --- Case 3: real ReAct, where an Observation comes from a tool --------------
# A fake production environment, standing in for the systems a real agent would
# query. The model cannot see any of this: it only gets what a tool returns.
FAKE_ENV = {
    "driver": "mysql",
    "tables": ["product_categories", "products", "orders"],
    "schema": {
        "products": [
            "id VARCHAR(36) NOT NULL PRIMARY KEY",
            "name VARCHAR(255) NOT NULL",
            "description TEXT",
            "price DECIMAL(10,2) NOT NULL",
            "stock INT NOT NULL DEFAULT 0",
            "created_at TIMESTAMP NOT NULL",
        ],
    },
    "logs": {
        "POST /products": [
            "Error during Exec: Error 1364: Field 'created_at' doesn't have a default value",
            "Error during Exec: Error 1364: Field 'created_at' doesn't have a default value",
        ],
    },
}


@tool
def get_database_driver() -> str:
    """Return the SQL driver the service connects with, e.g. mysql or postgres."""
    return FAKE_ENV["driver"]


@tool
def get_table_schema(table: str) -> str:
    """Return the column definitions of a table, one per line."""
    columns = FAKE_ENV["schema"].get(table)
    if not columns:
        known = ", ".join(FAKE_ENV["tables"])
        return f"Table '{table}' not found. Existing tables: {known}"
    return "\n".join(columns)


@tool
def get_recent_logs(endpoint: str) -> str:
    """Return the most recent server log lines for an endpoint, e.g. "POST /products"."""
    lines = FAKE_ENV["logs"].get(endpoint)
    if not lines:
        return f"No recent log lines for '{endpoint}'"
    return "\n".join(lines)


msg3 = """
You are a Go backend engineer debugging a REST API.
A user reports that `POST /products` always returns HTTP 500.

The handler runs this statement, and returns 500 whenever Exec fails:

    INSERT INTO products (id, name, description, price, stock) VALUES (?, ?, ?, ?, ?)

You have tools available. Use them to find the cause instead of guessing: state a
short Thought, then call a tool, then read its result before the next Thought.
When you know the cause, stop calling tools and answer with "Final Answer:" plus
the fix.
"""

MAX_STEPS = 6

# temperature=0 keeps runs comparable, so differences come from the prompt
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatOpenAI(model="gpt-5-nano") # reasoning model: does not accept a custom temperature

prompts = [msg1, msg2]

# The calls are independent, so batch runs them in parallel
for prompt, response in zip(prompts, llm.batch(prompts)):
    print_llm_result(prompt, response)

# Case 3 is a loop, not a single call: the model asks for a tool, the code runs it
# and appends the real result, and the model reasons again over what came back.
tools = [get_database_driver, get_table_schema, get_recent_logs]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage(msg3)]
tool_calls_made = 0
console.print(Text("REACT LOOP (real tools):", style="bold green"))

for step in range(1, MAX_STEPS + 1):
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    if ai_message.content:
        console.print(Text(f"\n[step {step}] Thought: ", style="bold green"), end="")
        console.print(Text(str(ai_message.content), style="bold blue"))

    # No tool call means the model is done reasoning and gave its final answer
    if not ai_message.tool_calls:
        break

    for call in ai_message.tool_calls:
        # This is the difference from msg1 and msg2: the Observation below is a
        # return value produced by code, not text the model wrote about itself
        observation = tools_by_name[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=observation, tool_call_id=call["id"]))
        tool_calls_made += 1

        console.print(Text(f"[step {step}] Action: ", style="bold green"), end="")
        console.print(Text(f"{call['name']}({call['args']})", style="bold yellow"))
        console.print(Text(f"[step {step}] Observation: ", style="bold green"), end="")
        console.print(Text(observation, style="bold blue"))
else:
    console.print(Text(f"\nStopped after {MAX_STEPS} steps without a final answer", style="bold red"))

console.print(f"\n[bold]Tool calls made:[/bold] {tool_calls_made}")
