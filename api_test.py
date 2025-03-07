import requests
import time
import base64
from PIL import Image
from io import BytesIO
import json

# Configuration
BASE_URL = "http://localhost:5000/api"
SAVE_IMAGES = True

def get_screen(save_image=False):
    """Get the current screen image"""
    response = requests.get(f"{BASE_URL}/screen?format=base64")
    data = response.json()
    
    if "image" in data:
        # Decode the base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_data))
        
        if save_image:
            timestamp = int(time.time())
            image.save(f"screen_{timestamp}.png")
            print(f"Saved screen to screen_{timestamp}.png")
        
        return image
    else:
        print("Error getting screen:", data)
        return None

def get_game_state():
    """Get the current game state"""
    response = requests.get(f"{BASE_URL}/state")
    return response.json()

def press_button(button, delay=1):
    """Press a button on the Game Boy"""
    response = requests.post(f"{BASE_URL}/button", json={
        "button": button,
        "delay": delay
    })
    return response.json()

def get_pokemon_location():
    """Get the current map location details"""
    response = requests.get(f"{BASE_URL}/pokemon/location")
    return response.json()

def get_pokemon_party():
    """Get detailed information about the Pokémon party"""
    response = requests.get(f"{BASE_URL}/pokemon/party")
    return response.json()

def save_game_state(path="save_test.state"):
    """Save the current game state"""
    response = requests.post(f"{BASE_URL}/save", json={"path": path})
    return response.json()

def load_game_state(path="save_test.state"):
    """Load a saved game state"""
    response = requests.post(f"{BASE_URL}/load", json={"path": path})
    return response.json()

def navigate_to(x, y, max_steps=100):
    """Navigate to a specific position on the current map"""
    response = requests.post(f"{BASE_URL}/move", json={
        "x": x,
        "y": y,
        "max_steps": max_steps
    })
    return response.json()

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def walk_around_demo():
    """Demo function to walk around the starting area"""
    # First press A a few times to get through intro
    for _ in range(5):
        press_button("a", 5)
        time.sleep(0.5)
    
    # Walk down
    press_button("down", 3)
    time.sleep(0.5)
    get_screen(save_image=True)
    
    # Walk right
    press_button("right", 3)
    time.sleep(0.5)
    get_screen(save_image=True)
    
    # Walk up
    press_button("up", 3)
    time.sleep(0.5)
    get_screen(save_image=True)
    
    # Walk left
    press_button("left", 3)
    time.sleep(0.5)
    get_screen(save_image=True)

def main():
    """Main function to demonstrate API usage"""
    
    # Get and save the initial screen
    initial_screen = get_screen(save_image=SAVE_IMAGES)
    
    # Get the initial game state
    initial_state = get_game_state()
    print("\nInitial game state:")
    print_json(initial_state)
    
    # Try to get the Pokémon location information
    location_info = get_pokemon_location()
    print("\nPokémon location information:")
    print_json(location_info)
    
    # Try to get the Pokémon party information
    party_info = get_pokemon_party()
    print("\nPokémon party information:")
    print_json(party_info)
    
    # Demo: Walk around
    print("\nStarting walk-around demo...")
    walk_around_demo()
    
    # Get the updated game state
    updated_state = get_game_state()
    print("\nUpdated game state:")
    print_json(updated_state)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()