# Manufacturing Agent MVP

A simple yet powerful agent for manufacturing operations that can handle both Q&A and workflow automation tasks.

## Features

- **Order Expediting Workflow**: Automate the process of expediting manufacturing orders
- **Document Q&A**: Answer general questions about company policies and procedures
- **Modular Design**: Easy to extend with new workflows and tools
- **REST API**: Simple HTTP interface for integration with other systems

## Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) LangSmith account for tracing

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd manufacturing-agent-mvp
   ```

2. Create and activate a virtual environment:
   ```powershell
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key.

## Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```

#### 2. Interact with the Agent
```http
POST /invoke_agent
Content-Type: application/json

{
    "query": "Can you help me expedite order URG-456?"
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
