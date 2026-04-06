"""
Start Ray Serve with explicit host binding to 0.0.0.0
so it's accessible from outside the Docker container.
"""
import ray
import time
from ray import serve

# Start Ray
ray.init()

# Configure HTTP to bind to all interfaces (not just localhost)
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

# Import and deploy the app
from app import app
serve.run(app)

print("Ray Serve is running on http://0.0.0.0:8000")

# Keep the process alive
while True:
    time.sleep(60)
