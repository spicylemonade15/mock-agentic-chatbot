from memory import EnhancedAgentOrchestrator
x= EnhancedAgentOrchestrator()
app = x._create_graph()
from IPython.display import Image, display


graph_image = app.get_graph().draw_mermaid_png()


display(Image(graph_image))

graph_image = app.get_graph().draw_mermaid_png()

with open("langgraph_visualization.png", "wb") as f:
    f.write(graph_image)

print("done")