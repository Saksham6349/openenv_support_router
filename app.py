import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(title="OpenEnv - Support Ticket Router")

# --- OpenEnv Pydantic Models ---

class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    queue_remaining: int
    task_level: str

class Action(BaseModel):
    department: str
    urgency: str

class Reward(BaseModel):
    value: float
    reason: str

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task: str = "easy"

# --- Tasks & Data ---

TICKETS = {
    "ticket_01": {"subject": "Forgot password", "body": "I can't login, reset link doesn't work.", "dept": "tech_support", "urgency": "high"},
    "ticket_02": {"subject": "Double charged", "body": "My credit card shows two charges for my monthly plan.", "dept": "billing", "urgency": "critical"},
    "ticket_03": {"subject": "Enterprise SSO", "body": "Does the enterprise package include SSO and SAML?", "dept": "sales", "urgency": "medium"},
    "ticket_04": {"subject": "Spam message", "body": "Buy cheap followers now! Click here.", "dept": "spam", "urgency": "low"},
    "ticket_05": {"subject": "Job Application", "body": "I submitted my resume last week. Any updates?", "dept": "hr", "urgency": "medium"},
}

# easy: 1 ticket
# medium: 3 tickets
# hard: 5 tickets
TASKS = {
    "easy": ["ticket_01"],
    "medium": ["ticket_01", "ticket_02", "ticket_03"],
    "hard": ["ticket_01", "ticket_02", "ticket_03", "ticket_04", "ticket_05"]
}

# Simple in-memory state
state_db = {
    "task": "easy",
    "queue": [],
    "total_reward": 0.0,
    "steps": 0,
    "max_steps": 10
}

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "environment": "OpenEnv Support Router"}

@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    task = req.task if req and req.task else "easy"
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from {list(TASKS.keys())}")
    
    state_db["task"] = task
    state_db["queue"] = TASKS[task].copy()
    state_db["total_reward"] = 0.0
    state_db["steps"] = 0
    state_db["max_steps"] = len(TASKS[task]) * 2  # max 2 attempts per ticket on average
    
    return _get_observation()

def _get_observation() -> Observation:
    if not state_db["queue"]:
        return Observation(
            ticket_id="None",
            subject="No active tickets",
            body="",
            queue_remaining=0,
            task_level=state_db["task"]
        )
    current_ticket = TICKETS[state_db["queue"][0]]
    return Observation(
        ticket_id=state_db["queue"][0],
        subject=current_ticket["subject"],
        body=current_ticket["body"],
        queue_remaining=len(state_db["queue"]),
        task_level=state_db["task"]
    )

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    if not state_db["queue"]:
        # Episode is already over
        return StepResponse(
            observation=_get_observation(),
            reward=Reward(value=0.0, reason="Episode already finished."),
            done=True,
            info={"score": 0.0}
        )
    
    state_db["steps"] += 1
    
    current_ticket_id = state_db["queue"][0]
    expected = TICKETS[current_ticket_id]
    
    dept_match = action.department.lower() == expected["dept"]
    urgency_match = action.urgency.lower() == expected["urgency"]
    
    reward_val = 0.0
    reasons = []
    
    if dept_match:
        reward_val += 0.5
        reasons.append("Correct department")
    else:
        reasons.append(f"Wrong department (expected {expected['dept']})")
        
    if urgency_match:
        reward_val += 0.5
        reasons.append("Correct urgency")
    else:
        reasons.append(f"Wrong urgency (expected {expected['urgency']})")
        
    # Agent properly evaluated the ticket
    state_db["queue"].pop(0)
    state_db["total_reward"] += reward_val
    
    done = len(state_db["queue"]) == 0 or state_db["steps"] >= state_db["max_steps"]
    
    # Calculate score if done (Score between 0.0 and 1.0)
    info = {}
    if done:
        total_possible = len(TASKS[state_db["task"]]) * 1.0
        score = state_db["total_reward"] / total_possible if total_possible > 0 else 0.0
        info["score"] = score
        info["total_reward"] = state_db["total_reward"]
        
    return StepResponse(
        observation=_get_observation(),
        reward=Reward(value=reward_val, reason=", ".join(reasons)),
        done=done,
        info=info
    )

@app.get("/state")
def state():
    return state_db
