import os
import time
import json
import shutil
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import BytesIO

from pyboy import PyBoy
from PIL import Image
import base64
import numpy as np

from pokemon_helper import PokemonHelper

# PokeAgent main class
class PokeAgent:
    def __init__(self, rom_path, save_state_path=None):
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.pyboy = PyBoy(rom_path)
        
        # Load save state if provided
        if save_state_path and os.path.exists(save_state_path):
            with open(save_state_path, "rb") as f:
                self.pyboy.load_state(f)
        
        # Initialize the game
        self.pyboy.tick()
        
        # Initialize Pokemon helper
        self.pokemon_helper = PokemonHelper(self.pyboy)
        
        # Memory address constants for Pokemon Red
        self.PLAYER_X = 0xD362
        self.PLAYER_Y = 0xD361
        self.MAP_ID = 0xD35E
        self.PARTY_SIZE = 0xD163
        self.BADGES = 0xD356
        self.CURRENT_HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        self.MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        self.POKEMON_LEVEL_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        self.POKEMON_SPECIES_ADDRESSES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        self.EVENT_FLAGS_START = 0xD747
        self.EVENT_FLAGS_END = 0xD87E
        
        # Last button pressed
        self.last_button = None
        
        # Load event names
        self.event_names = {}
        event_json_path = "events.json"
        if os.path.exists(event_json_path):
            with open(event_json_path, "r") as f:
                self.event_names = json.load(f)
    
    def tick(self, n=1):
        """Process n frames"""
        return self.pyboy.tick(n)
    
    def button(self, input_key, delay=1):
        """Press a button and release after delay ticks"""
        valid_buttons = ["a", "b", "start", "select", "left", "right", "up", "down"]
        if input_key not in valid_buttons:
            raise ValueError(f"Invalid input: {input_key}. Must be one of {valid_buttons}")
        
        self.last_button = input_key
        self.pyboy.button(input_key)
        self.pyboy.tick(delay)
        
        return True
    
    def get_screen_image(self):
        """Get the current screen as a PIL Image"""
        screen_image = self.pyboy.screen.image.copy()
        return screen_image
    
    def get_screen_base64(self):
        """Get the current screen as a base64 encoded string"""
        image = self.get_screen_image()
        buffered = BytesIO()
        image = image.resize((320, 288), Image.NEAREST)  # Upscaling for better visibility
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def get_player_position(self):
        """Get the player's position and current map"""
        x = self.pyboy.memory[self.PLAYER_X]
        y = self.pyboy.memory[self.PLAYER_Y]
        map_id = self.pyboy.memory[self.MAP_ID]
        return {"x": x, "y": y, "map_id": map_id}
    
    def read_hp(self, start_address):
        """Read HP value (2 bytes) from memory"""
        return 256 * self.pyboy.memory[start_address] + self.pyboy.memory[start_address + 1]
    
    def get_party_info(self):
        """Get information about the Pokémon party"""
        party_size = self.pyboy.memory[self.PARTY_SIZE]
        pokemon = []
        
        for i in range(min(party_size, 6)):
            species_id = self.pyboy.memory[self.POKEMON_SPECIES_ADDRESSES[i]]
            level = self.pyboy.memory[self.POKEMON_LEVEL_ADDRESSES[i]]
            current_hp = self.read_hp(self.CURRENT_HP_ADDRESSES[i])
            max_hp = self.read_hp(self.MAX_HP_ADDRESSES[i])
            
            pokemon.append({
                "species_id": species_id,
                "level": level,
                "current_hp": current_hp,
                "max_hp": max_hp,
                "hp_percent": current_hp / max(max_hp, 1) * 100
            })
        
        return {
            "party_size": party_size,
            "pokemon": pokemon
        }
    
    def get_badges(self):
        """Get the badges the player has obtained"""
        badges_byte = self.pyboy.memory[self.BADGES]
        badges = []
        badge_names = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
        
        for i in range(8):
            if (badges_byte >> i) & 1:
                badges.append(badge_names[i])
        
        return {
            "badges_byte": badges_byte,
            "badges": badges,
            "count": len(badges)
        }
    
    def read_bit(self, addr, bit):
        """Read a specific bit from a memory address"""
        return bin(256 + self.pyboy.memory[addr])[-bit - 1] == "1"
    
    def get_event_flags(self):
        """Get all event flags that are set"""
        event_flags = {}
        
        for address in range(self.EVENT_FLAGS_START, self.EVENT_FLAGS_END):
            val = self.pyboy.memory[address]
            for idx in range(8):
                if (val >> idx) & 1:
                    key = f"0x{address:X}-{idx}"
                    flag_name = self.event_names.get(key, "Unknown")
                    event_flags[key] = flag_name
        
        return event_flags
    
    def get_game_state(self):
        """Get a comprehensive game state"""
        return {
            "position": self.get_player_position(),
            "party": self.get_party_info(),
            "badges": self.get_badges(),
            "events": self.get_event_flags(),
            "last_button": self.last_button
        }
    
    def move_to_location(self, target_x, target_y, max_steps=100):
        """Try to move to a specific location on the current map"""
        steps_taken = 0
        current_pos = self.get_player_position()
        
        while (current_pos["x"] != target_x or current_pos["y"] != target_y) and steps_taken < max_steps:
            # Choose direction based on current position
            if current_pos["x"] < target_x:
                self.button("right")
            elif current_pos["x"] > target_x:
                self.button("left")
            elif current_pos["y"] < target_y:
                self.button("down")
            elif current_pos["y"] > target_y:
                self.button("up")
            
            # Update position and counter
            steps_taken += 1
            current_pos = self.get_player_position()
        
        return steps_taken < max_steps  # Return True if we reached the destination
    
    def save_state(self, path=None):
        """Save the current game state"""
        save_path = path or self.save_state_path or "save_state.state"
        with open(save_path, "wb") as f:
            self.pyboy.save_state(f)
        return save_path
    
    def load_state(self, path=None):
        """Load a game state"""
        load_path = path or self.save_state_path
        if load_path and os.path.exists(load_path):
            with open(load_path, "rb") as f:
                self.pyboy.load_state(f)
            return True
        return False
    
    def close(self):
        """Close the PyBoy instance"""
        if self.pyboy:
            self.pyboy.stop()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global PokeAgent instance
agent = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/controller')
def controller():
    return render_template('controller.html')

@app.route('/api/init', methods=['POST'])
def init_agent():
    global agent
    
    data = request.json
    rom_path = data.get('rom_path', 'PokemonRed.gb')
    save_state_path = data.get('save_state_path')
    
    if agent:
        agent.close()
    
    agent = PokeAgent(rom_path, save_state_path)
    return jsonify({"status": "success", "message": "PokeAgent initialized"})

@app.route('/api/screen', methods=['GET'])
def get_screen():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    format_type = request.args.get('format', 'base64')
    
    if format_type == 'base64':
        img_data = agent.get_screen_base64()
        return jsonify({"image": img_data})
    else:
        img = agent.get_screen_image()
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

@app.route('/api/button', methods=['POST'])
def press_button():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    data = request.json
    button = data.get('button')
    delay = data.get('delay', 1)
    
    try:
        result = agent.button(button, delay)
        # Emit the button press event to all connected clients
        socketio.emit('button_press', {'button': button})
        return jsonify({"status": "success", "button": button, "result": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/state', methods=['GET'])
def get_state():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.get_game_state())

@app.route('/api/position', methods=['GET'])
def get_position():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.get_player_position())

@app.route('/api/party', methods=['GET'])
def get_party():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.get_party_info())

@app.route('/api/badges', methods=['GET'])
def get_badges():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.get_badges())

@app.route('/api/events', methods=['GET'])
def get_events():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.get_event_flags())

@app.route('/api/move', methods=['POST'])
def move_to_location():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    data = request.json
    target_x = data.get('x')
    target_y = data.get('y')
    max_steps = data.get('max_steps', 100)
    
    if target_x is None or target_y is None:
        return jsonify({"error": "Target coordinates required"}), 400
    
    # Use the Pokemon Helper for navigation if available
    if agent.pokemon_helper.is_game_loaded():
        result = agent.pokemon_helper.navigate_to(target_x, target_y, max_steps)
        return jsonify(result)
    else:
        result = agent.move_to_location(target_x, target_y, max_steps)
        return jsonify({
            "status": "success" if result else "failure",
            "reached_destination": result,
            "current_position": agent.get_player_position()
        })

@app.route('/api/save', methods=['POST'])
def save_state():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    data = request.json
    path = data.get('path')
    
    save_path = agent.save_state(path)
    return jsonify({"status": "success", "save_path": save_path})

@app.route('/api/load', methods=['POST'])
def load_state():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    data = request.json
    path = data.get('path')
    
    result = agent.load_state(path)
    return jsonify({"status": "success" if result else "failure"})

@app.route('/api/tick', methods=['POST'])
def tick():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    data = request.json
    n = data.get('n', 1)
    
    agent.tick(n)
    return jsonify({"status": "success", "ticks": n})

# Pokemon-specific API endpoints
@app.route('/api/pokemon/location', methods=['GET'])
def get_pokemon_location():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.pokemon_helper.get_map_location())

@app.route('/api/pokemon/party', methods=['GET'])
def get_pokemon_party():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify(agent.pokemon_helper.get_detailed_party())

@app.route('/api/pokemon/tiles', methods=['GET'])
def get_pokemon_tiles():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    return jsonify({
        "walkable": agent.pokemon_helper.get_walkable_tiles(),
        "tiles": agent.pokemon_helper.get_tiles()
    })

@app.route('/api/pokemon/maps', methods=['GET'])
def get_pokemon_maps():
    if not agent:
        return jsonify({"error": "PokeAgent not initialized"}), 400
    
    if not hasattr(agent.pokemon_helper, 'map_data'):
        return jsonify({"error": "Map data not available"}), 400
    
    return jsonify(agent.pokemon_helper.map_data)

@socketio.on('connect')
def handle_connect():
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    pass

# Ensure agent is properly closed when application exits
import atexit

@atexit.register
def cleanup():
    global agent
    if agent:
        agent.close()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)