# 🏭 Manufacturing Agent MVP

A production-ready AI agent for manufacturing operations, supporting both Q&A and workflow automation with Google Gemini as the LLM backend.

## ✨ Features

- **LLM Backend**: Powered by Google Gemini
- **Order Management**: Automate manufacturing order workflows and expediting
- **Document Intelligence**: Advanced Q&A for policies, procedures, and documentation
- **Modular Architecture**: Easily extensible with custom tools and workflows
- **RESTful API**: Robust HTTP interface for system integration
- **Secure Configuration**: Environment-based settings with sensible defaults
- **Production Ready**: Logging, error handling, and monitoring support
- **Containerized**: Docker support for easy deployment
- **Reproducible Builds**: Pinned dependencies with pip-tools

## 📊 Evaluation System

The Manufacturing Agent includes a robust evaluation system to monitor and improve LLM response quality. The system automatically evaluates every LLM response using LangSmith and custom evaluators.

### 🔑 Environment Variables

Add these to your `.env` file for evaluation:

```
# Evaluation Configuration
EVAL_ENABLED=true
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=manufacturing-agent-evals
LANGCHAIN_API_KEY=your_langsmith_api_key

# Optional: Override default evaluation model
EVAL_GEMINI_MODEL=gemini-pro
EVAL_GEMINI_TEMPERATURE=0.3
EVAL_GEMINI_MAX_TOKENS=2048
```

### 🛠 Running Evaluations

1. **Automatic Evaluation**
   - Enabled by default for all LLM responses when `EVAL_ENABLED=true`
   - Runs asynchronously in the background to avoid latency

2. **Manual Evaluation**
   ```python
   from src.evaluator import ResponseEvaluator
   
   evaluator = ResponseEvaluator()
   
   # Evaluate a response
   eval_result = await evaluator.evaluate_response(
       inputs={"question": "Sample question"},
       outputs={"answer": "Sample answer"}
   )
   ```

3. **Viewing Evaluation Results**
   - Access metrics via API: `GET /evaluation/metrics`
   - View detailed traces in LangSmith dashboard

### 📈 Evaluation Metrics

Key metrics tracked:
- **Correctness**: Factual accuracy of responses
- **Completeness**: Coverage of required information
- **Relevance**: Appropriateness to the query
- **Helpfulness**: Practical utility of the response

### 🧪 Testing Evaluation

Run the test script:
```bash
python -m pytest tests/test_evaluation.py -v
```

## 🚀 Quick Start

### Prerequisites

- Docker (recommended)
- OR Python 3.11+ with pip
- [Google AI Studio API Key](https://makersuite.google.com/app/apikey)

## 🐳 Docker Setup

### Option 1: Docker Compose (Recommended for Development)

The easiest way to get started with development is using Docker Compose:

1. **Create environment file**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your API keys and configuration.

2. **Start the development server**:
   ```bash
   docker-compose up --build
   ```
   This will:
   - Build the development image with hot-reload enabled
   - Mount your source code as a volume for live updates
   - Start the server on `http://localhost:8000`

3. **View logs**:
   ```bash
   docker-compose logs -f
   ```

4. **Run tests**:
   ```bash
   docker-compose exec app pytest
   ```

### Option 2: Manual Docker Build (Production)

1. **Generate locked requirements files**:
   ```bash
   pip install pip-tools
   pip-compile --generate-hashes requirements-core.in --output-file=requirements-core.txt
   pip-compile --generate-hashes requirements-rag.in --output-file=requirements-rag.txt
   pip-compile --generate-hashes requirements-dev.in --output-file=requirements-dev.txt
   ```

2. **Build the production image**:
   ```bash
   docker build -t manufacturing-agent .
   ```

3. **Run the container**:
   ```bash
   docker run -p 8000:8000 --env-file .env manufacturing-agent
   ```

   The API will be available at `http://localhost:8000`

### Development vs Production

- **Development**: Uses `docker-compose.yml` with hot-reload enabled
- **Production**: Uses multi-stage Dockerfile with optimized builds

## 🛠 Development Workflow

### Prerequisites
- Python 3.11+
- pip-tools (`pip install pip-tools`)
- Docker and Docker Compose (recommended)

### Development with Docker (Recommended)

1. Start the development environment:
   ```bash
   docker-compose up --build
   ```
2. The server will automatically reload when you make changes to the code.
3. Access the API at `http://localhost:8000`
4. Run tests:
   ```bash
   docker-compose exec app pytest
   ```

### Local Development (Without Docker)

### Setup

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
   # Install pip-tools if not already installed
   pip install pip-tools
   
   # Generate locked requirements files
   pip-compile --generate-hashes requirements-core.in --output-file=requirements-core.txt
   pip-compile --generate-hashes requirements-rag.in --output-file=requirements-rag.txt
   
   # Install dependencies
   pip install -r requirements-core.txt -r requirements-rag.txt
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
   
   # Model Configuration
   MODEL_PROVIDER=gemini
   
   # Gemini Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-pro
   GEMINI_TEMPERATURE=0.73
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

### 🌐 API Documentation

Once the development server is running, you can access:

- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative API Docs**: `http://localhost:8000/redoc`

## 🔍 Development Tips

### Running Tests
```bash
# Run all tests
docker-compose exec app pytest

# Run specific test file
docker-compose exec app pytest tests/test_module.py

# Run with coverage
docker-compose exec app pytest --cov=app tests/
```

### Code Quality
```bash
# Format code with black
docker-compose exec app black .

# Check code style with flake8
docker-compose exec app flake8

# Type checking with mypy
docker-compose exec app mypy .
```

### Managing Dependencies
- Add new packages to the appropriate `.in` file
- Regenerate requirements files:
  ```bash
  docker-compose exec app pip-compile --generate-hashes requirements-core.in
  docker-compose exec app pip-compile --generate-hashes requirements-rag.in
  docker-compose exec app pip-compile --generate-hashes requirements-dev.in
  ```

## 📚 Documentation
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

## 🤖 Supported Models

This project uses Google Gemini as the LLM provider.

### Google Gemini
- gemini-pro (recommended)

### Configuration
- Set `GEMINI_API_KEY` to your Google AI Studio API key

## Configuration

Copy `.env.example` to `.env` and update the following variables:

### Model Selection
- `MODEL_PROVIDER`: 'gemini' (default: 'gemini')

### Model Configuration

You can configure the following model settings in your `.env` file:

### Gemini Settings
- `GEMINI_API_KEY`: Your Google AI Studio API key (required)
- `GEMINI_MODEL`: The Gemini model to use (default: `gemini-pro`)
- `GEMINI_TEMPERATURE`: Controls randomness (0.0 to 1.0, default: 0.7)

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
