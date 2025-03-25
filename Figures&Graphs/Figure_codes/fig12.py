from graphviz import Digraph

# Create a directed graph
workflow = Digraph(format='png')

# Define nodes
workflow.node("A", "Job Posting")
workflow.node("B", "Preprocessing")
workflow.node("C", "BERT Embedding")
workflow.node("D", "Hybrid Model")
workflow.node("E", "Output (Real/Fake)")
workflow.node("F", "Moderation Action")

# Define edges
workflow.edge("A", "B")
workflow.edge("B", "C")
workflow.edge("C", "D")
workflow.edge("D", "E")
workflow.edge("E", "F")

# Render and display the flowchart
workflow.render("proposed_system_workflow", view=True)
