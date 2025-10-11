import ray
from ray import serve
from app import graph

ray.init(address="auto")

# expose on all interfaces, port 8000
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

serve.run(graph, route_prefix="/")
print("Serve app started at http://0.0.0.0:8000/")
