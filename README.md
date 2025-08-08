# GitHub Models - Telegram Bot

A conversational Telegram chatbot using GitHub AI models, including the latest GPT-5 models.

## Supported Models

The bot supports various AI models including:

### OpenAI GPT Models
- **GPT-4.1** (default) - `openai/gpt-4.1`
- **GPT-4.1 Mini** - `openai/gpt-4.1-mini` 
- **GPT-4.1 Nano** - `openai/gpt-4.1-nano`
- **GPT-5** - `openai/gpt-5` (new)
- **GPT-5 Mini** - `openai/gpt-5-mini` (new)
- **GPT-5 Nano** - `openai/gpt-5-nano` (new) 
- **GPT-5 Chat** - `openai/gpt-5-chat` (new)
- **O3** - `openai/o3`
- **O4 Mini** - `openai/o4-mini`

### Other Models
- DeepSeek V3 and R1
- Microsoft MAI-DS-R1
- xAI Grok-3 and Grok-3 Mini

## Usage

Users can specify which model to use with the `use:` directive:

```
use: gpt5 Tell me about quantum computing
use: gpt-5-mini Summarize this article  
use: gpt5chat Help me write an email
```

Use `/models` command to see all available models.

**Note:** GPT-4.1 remains the default model for backwards compatibility.
