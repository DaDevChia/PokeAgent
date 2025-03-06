import json
import shutil
from pathlib import Path

# Copy the events.json file from v2 directory
source_path = Path("v2/events.json")
if source_path.exists():
    with open(source_path, "r") as f:
        events_data = json.load(f)
    
    with open("events.json", "w") as f:
        json.dump(events_data, f, indent=2)
    print("events.json copied successfully")
else:
    print("Warning: v2/events.json not found")

# Copy the map_data.json file from v2 directory
source_path = Path("v2/map_data.json")
if source_path.exists():
    with open(source_path, "r") as f:
        map_data = json.load(f)
    
    with open("map_data.json", "w") as f:
        json.dump(map_data, f, indent=2)
    print("map_data.json copied successfully")
else:
    print("Warning: v2/map_data.json not found")

print("Setup complete!")