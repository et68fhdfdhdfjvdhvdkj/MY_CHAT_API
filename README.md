# Coding Tutor API

A small FastAPI service that exposes a chat endpoint for coding help and runs either against GitHub Models or a mock fallback mode.

## Features

- Chat endpoint for coding questions
- Streaming chat endpoint for event-stream responses
- Built-in mock mode for offline or demo usage
- Configurable CORS origins for frontend apps
- Health and config endpoints for deployment checks

## Quick start

1. Create a virtual environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Copy the sample environment file and set your values:

   cp .env.example .env

3. Start the app:

   python main.py

4. Visit:

   - http://localhost:8000/
   - http://localhost:8000/chat
   - http://localhost:8000/api/config

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| HOST | No | Bind host. Default: 0.0.0.0 |
| PORT | No | Port to run on. Default: 8000 |
| USE_MOCK_MODE | No | Set to true to disable model calls and use the mock response |
| GITHUB_TOKEN | Optional if mock mode is enabled | GitHub token used for GitHub Models |
| GH_TOKEN | Optional fallback | Another supported token name |
| GITHUB_MODELS_MODEL | No | Model to call. Default: openai/gpt-4o-mini |
| GITHUB_MODELS_BASE_URL | No | Base URL for the model provider |
| CORS_ORIGINS | No | Comma-separated list of allowed frontend origins |

## Example

```bash
export USE_MOCK_MODE=false
export GITHUB_TOKEN=your_token_here
export GITHUB_MODELS_MODEL=openai/gpt-4o-mini
export CORS_ORIGINS=http://localhost:3000,https://my-app.example.com
python main.py
```

## API endpoints

- GET / => health information
- GET /api/test => simple API check
- GET /api/config => runtime configuration
- GET /chat => HTML chat UI
- POST /api/chat => single-response chat
- POST /api/chat/stream => streaming chat response

## Notes

- If no token is present and mock mode is off, the app returns a safe fallback message instead of failing.
- The app currently supports GitHub Models and OpenAI-compatible credentials via the environment.
