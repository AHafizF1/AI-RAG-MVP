# 🏭 Manufacturing Agent MVP

A production-ready AI agent for manufacturing operations, supporting both Q&A and workflow automation with multiple LLM backends including OpenAI and Google Gemini.

## ✨ Features

- **Multiple LLM Backends**: Seamless integration with OpenAI and Google Gemini
- **Order Management**: Automate manufacturing order workflows and expediting
- **Document Intelligence**: Advanced Q&A for policies, procedures, and documentation
- **Modular Architecture**: Easily extensible with custom tools and workflows
- **RESTful API**: Robust HTTP interface for system integration
- **Secure Configuration**: Environment-based settings with sensible defaults
- **Production Ready**: Logging, error handling, and monitoring support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Python virtual environment (recommended)
- API Key for your preferred LLM provider:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Google AI Studio API Key](https://makersuite.google.com/app/apikey)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/manufacturing-agent-mvp.git
   cd manufacturing-agent-mvp
   ```

2. **Set up a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your configuration:
   ```
   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   DEBUG=false
   
   # Model Selection (openai or gemini)
   MODEL_PROVIDER=gemini
   
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_TEMPERATURE=0.3
   
   # Gemini Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.0-flash-lite-preview
   GEMINI_TEMPERATURE=0.3
   ```

## 🏃‍♂️ Running the Application

### Development Mode
```bash
uvicorn main:app --reload
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation
Once running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🛠️ Development

### Project Structure
```
manufacturing-agent-mvp/
├── .github/             # GitHub workflows and templates
├── app/                  # Application source code
│   ├── api/             # API routes
│   ├── core/            # Core functionality
│   ├── models/          # Data models
│   └── services/        # Business logic
├── tests/               # Test suite
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore rules
├── main.py              # Application entry point
├── README.md            # This file
└── requirements.txt     # Project dependencies
```

### Testing
Run the test suite:
```bash
pytest
```

### Code Quality
```bash
# Linting
flake8 .

# Type checking
mypy .

```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with FastAPI and Pydantic
- Powered by OpenAI and Google Gemini
- Inspired by modern AI agent architectures
   - For OpenAI: Set `OPENAI_API_KEY`
   - For Gemini: Set `GEMINI_API_KEY`
   - Set `MODEL_PROVIDER` to either 'openai' or 'gemini'

## Configuration

Copy `.env.example` to `.env` and update the following variables:

### Model Selection
- `MODEL_PROVIDER`: Either 'openai' or 'gemini' (default: 'gemini')

### OpenAI Configuration (required if using OpenAI)
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: 'gpt-3.5-turbo')
- `OPENAI_TEMPERATURE`: 0.0 to 1.0 (default: 0.3)

### Gemini Configuration (required if using Gemini)
- `GEMINI_API_KEY`: Your Google API key with Vertex AI enabled
- `GEMINI_MODEL`: Model to use (default: 'gemini-1.5-pro')
- `GEMINI_TEMPERATURE`: 0.0 to 1.0 (default: 0.3)

### Feature Flags
- `ENABLE_MES_INTEGRATION`: Enable Manufacturing Execution System integration
- `ENABLE_ERP_INTEGRATION`: Enable Enterprise Resource Planning integration
- `ENABLE_SLACK_NOTIFICATIONS`: Enable Slack notifications

### Slack Configuration (optional)
- `SLACK_WEBHOOK_URL`: Webhook URL for Slack notifications
- `SLACK_CHANNEL`: Channel for notifications (e.g., '#manufacturing-alerts')

## Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /api/chat`: Process a natural language query
  ```json
  {
    "query": "What is the status of order ORD-123?",
    "conversation_id": "optional-conversation-id"
  }
  ```
  
  Example response:
  ```json
  {
    "response": "The status of order ORD-123 is 'In Production'.",
    "conversation_id": "optional-conversation-id",
    "metadata": {
      "model_provider": "gemini",
      "model": "gemini-1.5-pro"
    }
  }
  ```

- `GET /api/health`: Health check endpoint
  ```json
  {
    "status": "healthy",
    "model_provider": "gemini",
    "model": "gemini-1.5-pro",
    "version": "0.2.0",
    "debug": false
  }
  ```

### Example Queries

1. **Expedite an Order**:
   ```
   "Please help me rush order URG-456"
   ```

2. **Ask a Question**:
   ```
   "What is your return policy?"
   ```
   
   ```
   "What are your customer service hours?"
   ```

## Project Structure

- `actions.py`: Contains mock functions that simulate interactions with external systems
- `workflows.py`: Defines business workflows that orchestrate multiple actions
- `main.py`: FastAPI application and agent setup
- `requirements.txt`: Project dependencies
- `.env.example`: Template for environment variables

## Testing

1. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

2. Test the API using curl or Postman:
   ```bash
   # Test health check
   curl http://localhost:8000/health
   
   # Test order expediting
   curl -X POST "http://localhost:8000/invoke_agent" \
   -H "Content-Type: application/json" \
   -d '{"query": "Can you help me expedite order URG-456?"}'
   ```

## Extending the Agent

To add new capabilities:

1. Add new action functions in `actions.py`
2. Create new workflow functions in `workflows.py` if needed
3. Add new tool functions in `main.py`
4. Update the tools list in the `setup_agent()` function

## License

MIT
