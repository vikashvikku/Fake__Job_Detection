from graphviz import Digraph

dot = Digraph(comment='Challenges in Fake Job Detection', format='png')

dot.attr(rankdir='LR', size='8,5', nodesep='0.5', ranksep='0.5')
dot.attr('node', shape='box', style='filled', color='white', fontname='Helvetica', fontsize='12')

dot.node('A', 'Data Scarcity')
dot.node('B', 'Feature Overlap')
dot.node('C', 'Dynamic Scam Patterns')
dot.node('D', 'Multilingual Complexity')

# Connecting them as interconnected challenges
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'A')  # Forms a loop to show interconnection

# Optional: Cross connections for complexity
dot.edge('A', 'C')
dot.edge('B', 'D')

# Render the graph
dot.render('fake_job_detection_challenges', view=True)
