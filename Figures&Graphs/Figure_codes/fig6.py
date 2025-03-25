from graphviz import Digraph

dot = Digraph(comment='Deep Learning Architecture for Fake Job Detection', format='png')
dot.attr(rankdir='TB', size='10')

# Nodes
dot.node('A', 'Data Collection', shape='box', style='filled', fillcolor='#AED6F1')
dot.node('B', 'Text Preprocessing', shape='box', style='filled', fillcolor='#AED6F1')
dot.node('C', 'Embedding Layer\n(Word2Vec / BERT)', shape='box', style='filled', fillcolor='#AED6F1')
dot.node('D', 'Deep Neural Network\n(LSTM / CNN / Transformer)', shape='box', style='filled', fillcolor='#85C1E9')
dot.node('E', 'Fake / Real', shape='box', style='filled', fillcolor='#58D68D')

# Edges
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')

# Render and save
dot.render('fake_job_detection_architecture', view=True)
