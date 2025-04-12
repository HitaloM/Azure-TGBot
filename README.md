# Telegram Chat Bot

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![AIOgram: 3.19+](https://img.shields.io/badge/AIOgram-3.19%2B-2CA5E0.svg)](https://github.com/aiogram/aiogram)

A powerful conversational AI chatbot for Telegram, built with Python, AIOgram, and Azure AI Inference. Process text and media messages intelligently, generating context-aware responses powered by large language models.

## ✨ Features

- **Intelligent Conversations** - Leverages Azure AI models for natural and contextual conversations
- **Memory & Context** - Maintains conversation history for coherent, contextually relevant responses
- **Multi-modal Support** - Handles text, images, and more message types
- **Multi-chat Support** - Works in private chats and group conversations
- **Command System** - Built-in commands for controlling the bot behavior
- **User Management** - Whitelist functionality and permission levels
- **Tool Integration** - Web search, GitHub data access, and more via extensible tool system
- **Middleware** - Queue system to handle high traffic situations
- **Database Backend** - Persistent storage with SQLite (easily extendable to other databases)

## 📋 Requirements

- Python 3.13 or higher
- [Azure](https://azure.microsoft.com) account with Azure AI Inference service
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/HitaloM/Azure-TGBot.git
cd Azure-TGBot
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Regular installation
pip install . -U

# Development installation
pip install -e ".[dev]"
```

### 4. Configure environment variables

Create or edit `data/config.env` with the following values:

```env
# Required Bot Settings
BOT_TOKEN=your_telegram_bot_token
AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=your_azure_endpoint_url

# Optional Settings
DEPLOYMENT_NAME=your_model_deployment_name
WHITELIST_ENABLED=true
SUDO_USERS=1234567,7654321
BING_SEARCH_API_KEY=your_bing_search_api_key
```

### 5. Run the bot

```bash
python -m bot
```

## 🏗️ Project Structure

```plaintext
src/
  ├── bot/                   # Main bot package
  │   ├── __main__.py        # Entry point
  │   ├── config.py          # Configuration management
  │   ├── database/          # Database models and connection
  │   ├── filters/           # Custom message filters
  │   ├── handlers/          # Command and message handlers
  │   ├── middlewares/       # Request processing middleware
  │   └── utils/             # Utility functions
  │       ├── chat/          # Chat processing utilities
  │       │   ├── client.py  # Azure AI client integration
  │       │   ├── context.py # Context management
  │       │   ├── history.py # Chat history
  │       │   └── tools/     # External tool integrations
  └── ...
```

## 📝 Usage

Once your bot is running, you can interact with it on Telegram by:

- Starting a private conversation
- Adding it to a group chat
- Using commands like:
  - `/ask [question]` - Ask the bot a question
  - `/reset` - Reset your conversation history
  - `/models` - List available models
  - `/whitelist [user_id]` - Add a user to the whitelist (admin only)

## 🛠️ Advanced Configuration

### Custom System Messages

Edit `data/system.txt` to customize the bot's personality and behavior guidelines.

### Database Management

The bot uses SQLite by default. The database file is located at `data/db.sqlite3`.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [AIOgram](https://github.com/aiogram/aiogram) for the excellent Telegram Bot framework
- [Azure AI](https://azure.microsoft.com/services/cognitive-services/) for providing the AI models
- All contributors and users of this project
