import requests
import json

# Configuration
BASE_URL = "http://localhost:5000/api"


def get_agent_thoughts():
    """Get all recorded agent thoughts"""
    response = requests.get(f"{BASE_URL}/agent/thoughts")
    return response.json()

def add_agent_thought(thought):
    """Add a new agent thought"""
    response = requests.post(f"{BASE_URL}/agent/thoughts", json={"thought": thought})
    return response.json()

def add_memory(memory):
    """Add a new memory"""
    response = requests.post(f"{BASE_URL}/agent/memory", json={"memory": memory})
    return response.json()

def get_memory():
    """Get memories"""
    response = requests.get(f"{BASE_URL}/agent/memory")
    return response.json()
    
def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def walk_around_demo():
    # Add a new agent thought
    thought_response = add_agent_thought("This is a test thought 2.")
    print("\nAdd agent thought response:")
    print_json(thought_response)
    
    # Get all agent thoughts
    thoughts = get_agent_thoughts()
    print("\nAgent thoughts:")
    print_json(thoughts)

    # Add a new memory
    memory_response = add_memory("This is a test memory.")

    # Get all memories
    memories = get_memory()
    print("\nMemories:")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    walk_around_demo()