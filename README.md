# PokeAgent: Pokemon Red AI LLM Challenge Platform

PokeAgent is a Flask-based web application that provides an API for controlling a Pokemon Red game using the PyBoy emulator. It allows AI agents, LLMs, or humans to interact with the game through a REST API and WebSocket interface.

## Features

- 🎮 Game visualization and controller in the browser
- 📱 Mobile-friendly controller interface
- 🤖 REST API for programmatic control
- 📊 Game state monitoring and analysis
- 🚀 Pokemon-specific helper functions
- 📍 Navigation and pathfinding support

## Prerequisites

- Python 3.8 or higher
- A legally obtained Pokemon Red ROM file (not included)
- An initial save state (optional but recommended)

### If you are using MacOS
- Install Glut and XQuartz
  ```bash
  brew install freeglut
  brew cask install xquartz
  ```
- Log out and log back in to apply the changes

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dadevchia/pokeagent.git
   cd pokeagent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy your Pokemon Red ROM to the project directory and name it `PokemonRed.gb`

4. If you have an initial save state, place it in the project directory as `init.state`

5. Run the setup script to copy required data files:
   ```bash
   python setup.py
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   - Main interface: http://localhost:5000/
   - Mobile controller: http://localhost:5000/controller

3. Click the "Initialize Game" button to start the emulator

## API Endpoints

### Basic Emulator Control

- `POST /api/button`: Press a button
- `GET /api/screen`: Get the current screen image
- `GET /api/state`: Get the complete game state

### Game State

- `GET /api/position`: Get player position
- `GET /api/party`: Get Pokemon party information
- `GET /api/badges`: Get badge information
- `GET /api/events`: Get event flags that are set
- `POST /api/save`: Save the current game state
- `POST /api/load`: Load a saved game state

### Pokemon-Specific

- `GET /api/pokemon/location`: Get detailed map location information
- `GET /api/pokemon/party`: Get detailed Pokemon party data with names and stats
- `GET /api/pokemon/tiles`: Get tile and walkable area information
- `GET /api/pokemon/maps`: Get all available map data
- `POST /api/move`: Navigate to a specific location on the current map

## API Example

### Press a button
```bash
curl -X POST http://localhost:5000/api/button \
  -H "Content-Type: application/json" \
  -d '{"button": "a", "delay": 1}'
```

### Get the current screen
```bash
curl -X GET http://localhost:5000/api/screen?format=base64
```

### Get the current screen png
```bash
curl -X GET http://localhost:5000/api/screen
```

### Get game state
```bash
curl -X GET http://localhost:5000/api/state
```

### Move to a location
```bash
curl -X POST http://localhost:5000/api/move \
  -H "Content-Type: application/json" \
  -d '{"x": 10, "y": 15, "max_steps": 100}'
```

## For AI/LLM Integration

PokeAgent is designed to be easily integrated with AI agents and Large Language Models (LLMs). The API provides all the necessary information for an agent to understand the game state and make informed decisions.

### Sample Python Code for AI Integration

```python
import requests
import base64
from PIL import Image
from io import BytesIO

BASE_URL = "http://localhost:5000/api"

# Initialize the game
requests.post(f"{BASE_URL}/init", json={"rom_path": "PokemonRed.gb"})

# Get the current screen
response = requests.get(f"{BASE_URL}/screen?format=base64")
screen_data = response.json()["image"]
screen_image = Image.open(BytesIO(base64.b64decode(screen_data)))

# Get game state
game_state = requests.get(f"{BASE_URL}/state").json()

# Make a decision based on the state and screen
# ... AI logic here ...

# Execute an action
requests.post(f"{BASE_URL}/button", json={"button": "a"})
```

## Project Structure

- `app.py`: Main Flask application
- `pokemon_helper.py`: Pokemon-specific helper functions
- `templates/`: HTML templates for web interface
- `static/`: Static assets for web interface
- `events.json`: Pokemon game event flags
- `map_data.json`: Map location data

## License

This project is for educational and research purposes only. Pokemon is a trademark of Nintendo/Game Freak.

## Acknowledgements

- [PyBoy](https://github.com/Baekalfen/PyBoy): Game Boy emulator in Python
- [Pokemon Red/Blue: Technical Documentation](https://github.com/pret/pokered)
- [v2 Pokemon Red Reinforcement Learning](https://github.com/pwhiddy/PokemonRedExperiments)