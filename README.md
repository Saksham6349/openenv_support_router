# OpenEnv: Customer Support Router

## Project Motivation
The goal of this OpenEnv environment is to assess an LLM agent's capability in a standard business operation: processing and prioritizing incoming customer support emails. Real-world support centers receive thousands of unstructured messages daily, which must be accurately routed to the correct department (e.g., Tech Support, Billing, HR) and triaged correctly based on urgency (e.g., Low, Medium, High, Critical).

Unlike games or toy problems, this environment tests the practical semantic reasoning and classification abilities of language models, which is directly applicable to enterprise AI use cases. By scoring partial responses (correct department or correct urgency), it provides a granular reward signal.

## Architecture
The environment is written in Python using FastAPI, satisfying the OpenEnv specification by surfacing typed `reset`, `step`, and `state` endpoints. It operates entirely as an HTTP REST API.

## Action & Observation Spaces

### Observation Space
A structured JSON representation of an incoming ticket:
```json
{
  "ticket_id": "<string identifier>",
  "subject": "<str>",
  "body": "<str>",
  "queue_remaining": <int>,
  "task_level": "<string task level>"
}
```

### Action Space
A structured JSON object for the agent's decision:
```json
{
  "department": "<str: tech_support, billing, sales, hr, spam>",
  "urgency": "<str: low, medium, high, critical>"
}
```

## Setup Instructions

### Local Execution (Docker)
1. Build the Docker container:
   ```bash
   docker build -t openenv_support_router .
   ```
2. Run the environment server container:
   ```bash
   docker run -d -p 7860:7860 openenv_support_router
   ```
3. Execute the baseline Agent (make sure your LLM env variables are set):
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-3.5-turbo"
   export HF_TOKEN="your-openai-or-hf-api-key"
   python inference.py
   ```

## Task Configurations

1. **Easy Task**: The agent receives a single tech support ticket with a clear issue. Perfect score expected for basic models.
2. **Medium Task**: The agent receives 3 different tickets, including a critical billing bug. Models must differentiate urgency and departments across a small multi-task batch.
3. **Hard Task**: The agent processes 5 tickets, adding ambiguous edge cases like a spam phishing email and a non-customer HR inquiry. This acts as a robust challenge against hallucination and incorrect routing.

## Rewards
Agents receive `+0.5` for correctly identifying the target department, and `+0.5` for correctly identifying the targeted urgency level. Scores linearly aggregate per step resulting in a scale from `0.0` to `1.0`.

## Baseline Reference Scores
When running the `inference.py` evaluator with `gpt-3.5-turbo` or equivalently sized frontier models:
- **Easy**: `1.0 / 1.0`
- **Medium**: `1.0 / 1.0`
- **Hard**: `1.0 / 1.0` (Though weaker models often fail on the spam edge case).
