from rich.console import Console
from rich.text import Text

console = Console()


def _content_to_str(content):
    """Convert message content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return str(content)


def _get_token_usage(response):
    """Return (input, output, total) tokens, or None if the provider reports none."""
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]

    # Fallback for OpenAI-style metadata
    usage = (getattr(response, "response_metadata", None) or {}).get("token_usage")
    if usage:
        return usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]

    return None


def print_llm_result(prompt, response):
    """
    Print LLM prompt, response and token usage with colored formatting
    """
    # Print prompt
    console.print(Text("USER PROMPT:", style="bold green"))
    console.print(Text(str(prompt), style="bold blue"), end="\n\n")

    # Print response
    console.print(Text("LLM RESPONSE:", style="bold green"))
    console.print(Text(_content_to_str(response.content), style="bold blue"), end="\n\n")

    # Print token usage
    tokens = _get_token_usage(response)
    if tokens:
        input_tokens, output_tokens, total_tokens = tokens
        console.print(f"[bold]Input tokens:[/bold] [bright_black]{input_tokens}[/bright_black]")
        console.print(f"[bold]Output tokens:[/bold] [bright_black]{output_tokens}[/bright_black]")
        console.print(f"[bold]Total tokens:[/bold] [bright_black]{total_tokens}[/bright_black]")
    else:
        console.print("[bright_black]Token usage not available[/bright_black]")
    console.print(f"[yellow]{'-' * 50}[/yellow]")
