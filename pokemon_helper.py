from pathlib import Path
import json
import numpy as np
from pyboy import PyBoy
# Import the base plugin class to check API
from pyboy.plugins.base_plugin import PyBoyPlugin

class PokemonHelper:
    """Helper class to interact with the Pokemon game specifically"""
    
    def __init__(self, pyboy_instance):
        """Initialize with an existing PyBoy instance"""
        self.pyboy = pyboy_instance
        self.wrapper = None
        
        # Try to initialize the wrapper with the proper API
        try:
            # First try importing to check if it exists
            from pyboy.plugins.game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
            
            # Check the class signature
            try:
                # New API (with 3 arguments)
                # Common pattern for PyBoy plugins is (pyboy, mb, game_wrapper)
                # where mb is the motherboard and game_wrapper is the parent wrapper
                self.wrapper = GameWrapperPokemonGen1(
                    self.pyboy, 
                    self.pyboy.mb, 
                    self.pyboy.game_wrapper()
                )
            except TypeError:
                # Fallback to trying with 2 arguments
                try:
                    self.wrapper = GameWrapperPokemonGen1(self.pyboy, self.pyboy.mb)
                except TypeError:
                    print("WARNING: Could not initialize Pokemon wrapper - incompatible API")
            
            # Check if wrapper was initialized and enabled
            if self.wrapper and not self.wrapper.enabled():
                self.wrapper = None
                print("WARNING: Pokemon Red/Blue game not detected!")
        except ImportError:
            print("WARNING: GameWrapperPokemonGen1 not available in this version of PyBoy")
        
        # Load map data for location information
        self.map_data = {}
        map_data_path = Path("map_data.json")
        if map_data_path.exists():
            with open(map_data_path, "r") as f:
                self.map_data = json.load(f)["regions"]
                self.map_data = {int(e["id"]): e for e in self.map_data if e["id"] != "-1"}
    
    def is_game_loaded(self):
        """Check if the Pokemon game is properly loaded"""
        return self.wrapper is not None and self.wrapper.enabled()
    
    def get_map_location(self):
        """Get information about the current map location"""
        if not self.is_game_loaded():
            return {"error": "Pokemon game not loaded"}
        
        x, y, map_n = self.get_player_position()
        
        map_info = {"id": map_n, "name": "Unknown"}
        
        if map_n in self.map_data:
            map_info["name"] = self.map_data[map_n]["name"]
            map_info["coordinates"] = self.map_data[map_n]["coordinates"]
            map_info["tile_size"] = self.map_data[map_n]["tileSize"]
        
        return {
            "position": {"x": x, "y": y},
            "map": map_info
        }
    
    def get_player_position(self):
        """Get the player's coordinates and map number"""
        return (
            self.pyboy.memory[0xD362],  # X position
            self.pyboy.memory[0xD361],  # Y position
            self.pyboy.memory[0xD35E]   # Map number
        )
    
    def get_walkable_tiles(self):
        """Get a matrix of walkable tiles on the current map"""
        if not self.is_game_loaded():
            return {"error": "Pokemon game not loaded"}
        
        try:
            walkable_matrix = self.wrapper._get_screen_walkable_matrix()
            return walkable_matrix.tolist()
        except Exception as e:
            return {"error": str(e)}
    
    def get_tiles(self):
        """Get the current tilemap grid"""
        if not self.is_game_loaded():
            return {"error": "Pokemon game not loaded"}
        
        try:
            screen_tiles = self.wrapper._get_screen_background_tilemap()
            return screen_tiles.tolist()
        except Exception as e:
            return {"error": str(e)}
    
    def get_pokemon_names(self):
        """Get a dictionary mapping Pokemon IDs to names"""
        # This is a simplified list - in a full implementation, you'd want all 151
        return {
            1: "Bulbasaur",
            2: "Ivysaur",
            3: "Venusaur",
            4: "Charmander",
            5: "Charmeleon",
            6: "Charizard",
            7: "Squirtle",
            8: "Wartortle",
            9: "Blastoise",
            25: "Pikachu",
            26: "Raichu",
            # Add more Pokemon name mappings as needed
        }
    
    def get_detailed_party(self):
        """Get detailed information about the Pokemon party"""
        if not self.is_game_loaded():
            return {"error": "Pokemon game not loaded"}
        
        party_count = self.pyboy.memory[0xD163]
        pokemon_names = self.get_pokemon_names()
        
        party = []
        for i in range(min(party_count, 6)):
            species_id = self.pyboy.memory[0xD164 + i]
            name = pokemon_names.get(species_id, f"Pokemon #{species_id}")
            
            level = self.pyboy.memory[0xD18C + i * 44]  # 44 bytes per party member
            
            # HP is stored as a 2-byte value
            current_hp = self.pyboy.memory[0xD16C + i * 44] * 256 + self.pyboy.memory[0xD16D + i * 44]
            max_hp = self.pyboy.memory[0xD18D + i * 44] * 256 + self.pyboy.memory[0xD18E + i * 44]
            
            # Get the moves (up to 4)
            moves = []
            for m in range(4):
                move_id = self.pyboy.memory[0xD168 + i * 44 + m]
                if move_id != 0:
                    moves.append(move_id)
            
            pokemon = {
                "position": i + 1,
                "species_id": species_id,
                "name": name,
                "level": level,
                "hp": {
                    "current": current_hp,
                    "max": max_hp,
                    "percent": (current_hp / max(max_hp, 1)) * 100
                },
                "moves": moves
            }
            
            party.append(pokemon)
        
        return {
            "count": party_count,
            "pokemon": party
        }
    
    def navigate_to(self, target_x, target_y, max_steps=100):
        """Navigate to a specific position on the current map"""
        if not self.is_game_loaded():
            return {"error": "Pokemon game not loaded"}
        
        steps_taken = 0
        x, y, map_id = self.get_player_position()
        
        while (x != target_x or y != target_y) and steps_taken < max_steps:
            # Determine direction to move
            if x < target_x:
                self.pyboy.button("right")
            elif x > target_x:
                self.pyboy.button("left")
            elif y < target_y:
                self.pyboy.button("down")
            elif y > target_y:
                self.pyboy.button("up")
            
            # Update position after moving
            self.pyboy.tick(1)
            x, y, new_map_id = self.get_player_position()
            
            # If map changed, stop navigation
            if new_map_id != map_id:
                return {
                    "success": False,
                    "reason": "Map changed during navigation",
                    "steps_taken": steps_taken,
                    "current_position": {"x": x, "y": y, "map": new_map_id}
                }
            
            steps_taken += 1
        
        return {
            "success": steps_taken < max_steps,
            "steps_taken": steps_taken,
            "current_position": {"x": x, "y": y, "map": map_id}
        }