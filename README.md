# Telegram Chat Bot

A chatbot developed in Python using AIOgram and Azure AI Inference to process text and media messages, generating intelligent responses through AI models.

## Introduction

This project aims to offer a robust chatbot for Telegram, supporting both direct messages and group chats. Built with Python and AIOgram, the bot integrates Azure AI Inference to access AI models that generate responses based on the conversation context.

## Features

- **Message Processing:** Handles and responds to text and media messages.
- **Azure AI Integration:** Uses Azure AI Inference to handle multiple AI models.
- **Telegram Support:** Compatible with private and group chats on Telegram.
- **Modularity:** An organized code structure that facilitates maintenance and project evolution.

## Setup and Installation

### Prerequisites

- Python 3.13 or higher.
- An [Azure](https://azure.microsoft.com) account with access to Azure AI Inference service.
- Telegram API Token (obtainable via [@BotFather](https://t.me/BotFather)).

### Installation

1. Clone the repository:

     ```bash
     git clone https://github.com/HitaloM/Azure-TGBot
     cd Azure-TGBot
     ```

2. Create and activate a virtual environment:

     ```bash
     python -m venv venv
     source venv/bin/activate      # Linux/macOS
     venv\Scripts\activate         # Windows
     ```

3. Install the dependencies:

     ```bash
     pip install . -U
     ```

4. Configure the environment variables in the `.env` file located at `data/config.env`:

     - `bot_token`: Telegram bot token.
     - `azure_api_key`: Credential for the Azure AI inference service.
     - `azure_endpoint`: URL for the Azure AI inference service endpoint.

5. Run the bot:

     ```bash
     python -m bot
     ```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
