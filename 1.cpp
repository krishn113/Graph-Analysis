# Required libraries for data handling and graph operations
import pandas as pd
import networkx as nx
import random

# Load Excel data into a pandas DataFrame
data = pd.read_excel('C:/dataset.xlsx')

# Create a directed graph to represent the relationships
graph = nx.DiGraph()

# Iterate through each row to add nodes and edges
for idx, row in data.iterrows():
    source = row.iloc[0]  # node representing the individual
    graph.add_node(source)
    connected = set()  # track edges to prevent repetition

    # Go through each of the 30 possible connections
    for i in range(1, 31):
        target = row.iloc[i]
        if pd.notna(target) and target not in ('', 'nan'):  # skip empty or invalid entries
            if (source, target) not in connected:
                graph.add_edge(source, target)
                connected.add((source, target))

# Simulate a random walk to estimate influence across the graph
def perform_random_walk(graph, steps=100000):
    all_nodes = list(graph.nodes)
    influence_count = {node: 0 for node in all_nodes}

    current = random.choice(all_nodes)  # starting point
    influence_count[current] += 1

    # Get outgoing edges from the current node
    outgoing = list(graph.out_edges(current))

    for _ in range(steps):
        if not outgoing:  # if no outgoing edges, jump to a random node
            current = random.choice(all_nodes)
        else:
            current = random.choice(outgoing)[1]  # move to a connected node

        influence_count[current] += 1
        outgoing = list(graph.out_edges(current))  # update the next possible moves

    return influence_count

# Sort nodes based on influence scores in descending order
def get_ranked_nodes(score_map):
    ranked_list = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in ranked_list]

# Execute random walk and display the most influential individual
scores = perform_random_walk(graph)
ranked_nodes = get_ranked_nodes(scores)
print("THE MOST INFLUENTIAL INDIVIDUAL IN THE NETWORK IS:", ranked_nodes[0])
