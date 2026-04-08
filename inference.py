import os
import requests
import json
import time
from openai import OpenAI

# API Configuration from Environment
api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
api_key = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

if not api_key:
    # Quick fallback for local testing if needed
    api_key = "dummy-key-if-not-set"

client = OpenAI(
    api_key=api_key,
    base_url=api_base_url,
)

tasks = ["easy", "medium", "hard"]
env_url = "http://localhost:7860"

def wait_for_env():
    # Wait until the env is up
    for _ in range(30):
        try:
            res = requests.get(env_url)
            if res.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def main():
    if not wait_for_env():
        print("Environment failed to start.")
        return

    for task in tasks:
        # Strict format: [START] task=...
        print(f"[START] task={task}")
        
        # Reset environment
        try:
            res = requests.post(f"{env_url}/reset", json={"task": task})
            res.raise_for_status()
            obs = res.json()
        except Exception as e:
            print(f"Error resetting env for task {task}: {e}")
            continue
        
        done = False
        total_reward = 0.0
        
        while not done:
            subject = obs.get("subject", "")
            body = obs.get("body", "")
            
            prompt = f"""You are an expert customer support dispatcher.
Route the given ticket to the correct department and assign urgency.
Departments available: tech_support, billing, sales, hr, spam.
Urgency levels: low, medium, high, critical.

Ticket Subject: {subject}
Ticket Body: {body}

Reply ONLY with a strictly valid JSON object. Do not include markdown formatting or backticks.
Format: {{"department": "sales", "urgency": "medium"}}
"""
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                reply = completion.choices[0].message.content.strip()
                
                # Cleanup markdown formatting just in case
                if reply.startswith("```json"):
                    reply = reply[7:-3].strip()
                if reply.startswith("```"):
                    reply = reply[3:-3].strip()
                    
                action = json.loads(reply)
            except Exception as e:
                # Default fallback action to keep moving
                action = {"department": "tech_support", "urgency": "medium"}
                
            # Formatting as required
            action_log = json.dumps(action, separators=(',', ':'))
            obs_log = json.dumps(obs, separators=(',', ':'))
            
            # Step the environment
            try:
                step_res = requests.post(f"{env_url}/step", json=action)
                step_res.raise_for_status()
                data = step_res.json()
                
                new_obs = data["observation"]
                reward_info = data["reward"]
                reward_val = reward_info["value"]
                done = data["done"]
                
                print(f"[STEP] action={action_log} observation={obs_log} reward={reward_val} done={done}")
                
                obs = new_obs
                total_reward += float(reward_val)
            except Exception as e:
                print(f"Error stepping env: {e}")
                break

        # Calculate final normalized score using info or total reward
        print(f"[END] total_reward={total_reward}")

if __name__ == "__main__":
    main()
