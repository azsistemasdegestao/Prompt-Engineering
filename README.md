# Prompt Engineering

Hands-on studies of prompt engineering techniques, implemented in Python with
[LangChain](https://python.langchain.com/) and the OpenAI models.

Each script is a small, self-contained experiment: it sends one or more prompts
to a model and prints the prompt, the response and the token usage side by side,
so the effect of a technique can be compared directly against a plain prompt.

## Repository structure

```
Prompt-Engineering/
└── 1-prompt-types/                   # Prompt types
    ├── 0-Role-prompting.py           # Role prompting (system persona)
    ├── 1-zero-shot.py                # Zero-shot prompting
    ├── 2-one-few-shot.py             # One-shot and few-shot prompting
    ├── 3-CoT.py                      # Chain of Thought
    ├── 3.1-CoT-Self-consistency.py   # CoT + self-consistency (majority vote)
    ├── 4-ToT.py                      # Tree of Thought
    ├── 5-SoT.py                      # Skeleton of Thought
    ├── 6-ReAct.py                    # ReAct (reasoning + acting with tools)
    ├── utils.py                      # Pretty printing of prompt, response and tokens
    └── requirements.txt
```

## Techniques covered

| Script | Technique | Idea |
| --- | --- | --- |
| `0-Role-prompting.py` | Role prompting | The same question answered under two different system personas, showing how the role shapes vocabulary and depth. |
| `1-zero-shot.py` | Zero-shot | Asking directly, with no examples — and how adding an output constraint tightens the answer. |
| `2-one-few-shot.py` | One-shot / few-shot | Teaching the output format with one example, then with several, for a log-severity classification task. |
| `3-CoT.py` | Chain of Thought | Asking the model to reason step by step before answering. Runs at `temperature=0` so the differences come from the prompt, not from sampling. |
| `3.1-CoT-Self-consistency.py` | CoT + self-consistency | Samples N independent reasoning paths at `temperature=0.7` and takes a majority vote over the final answers, in code rather than asking the model. |
| `4-ToT.py` | Tree of Thought | Generating several branches, evaluating each one, and selecting the best. Also shows why hiding the tree only works on a reasoning model. |
| `5-SoT.py` | Skeleton of Thought | Stage 1 produces a skeleton, stage 2 expands each point in its own parallel call — where the latency gain of the technique actually comes from. |
| `6-ReAct.py` | ReAct | Alternating `Thought` / `Action` / `Observation`, first simulated inside the prompt, then for real with a tool the code actually executes. |

## Requirements

- Python 3.11+
- An OpenAI API key

## Setup

```bash
git clone https://github.com/azsistemasdegestao/Prompt-Engineering.git
cd Prompt-Engineering/1-prompt-types

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create the `.env` file from the example and fill in your key:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is ignored by git and must never be committed.

## Running

Run any script from inside `1-prompt-types/`:

```bash
python 1-zero-shot.py
python 3-CoT.py
python 6-ReAct.py
```

Every run prints the prompt, the model response and the token usage, so the cost
of a technique is visible next to its benefit.

> **Note:** running these scripts calls the OpenAI API and consumes credits from
> your account.

## Dependencies

- `langchain-core` / `langchain-openai` — model interface and prompt templates
- `python-dotenv` — loads the API key from `.env`
- `rich` — colored terminal output

## License

Released under the [MIT License](LICENSE).
