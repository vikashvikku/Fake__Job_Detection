from graphviz import Digraph

# Create a directed graph
dot = Digraph(format='png')

# Define nodes
dot.node('D', 'Input Data', shape='parallelogram', style='filled', fillcolor='lightgrey')
dot.node('RF', 'Random Forest', shape='box', style='filled', fillcolor='lightblue')
dot.node('XGB1', 'XGBoost', shape='box', style='filled', fillcolor='lightblue')
dot.node('Meta', 'XGBoost Meta-Learner', shape='box', style='filled', fillcolor='lightgreen')
dot.node('P', 'Final Prediction', shape='ellipse', style='filled', fillcolor='lightyellow')

# Define edges (data flow)
dot.edge('D', 'RF')
dot.edge('D', 'XGB1')
dot.edge('RF', 'Meta')
dot.edge('XGB1', 'Meta')
dot.edge('Meta', 'P')

# Render the graph
dot.render('hybrid_model_architecture', view=True)
