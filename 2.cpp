# Import essential libraries for data processing and network analysis
import pandas as pd
import networkx as nx
import numpy as np

# Load dataset from Excel file into a DataFrame
data = pd.read_excel('C:/Users/nares/OneDrive/Desktop/cs101/project2/dataset simplified.xlsx')

# Initialize a directed graph structure
network = nx.DiGraph()

# Construct the graph: Add nodes and directed edges based on the dataset
for idx, row in data.iterrows():
    origin = row.iloc[0]
    network.add_node(origin)
    seen_edges = set()  # track unique edges to prevent redundancy
    for col in range(1, 31):  # columns 1 to 30 represent interactions
        target = row.iloc[col]
        if pd.notna(target) and target not in ('', 'nan'):
            if (origin, target) not in seen_edges:
                network.add_edge(origin, target)
                seen_edges.add((origin, target))
# Convert graph into an adjacency matrix
adj_matrix = nx.adjacency_matrix(network).toarray()

# Estimate potential missing connections using linear approximation
def estimate_links_via_matrix(adj_matrix, threshold):
    prediction_flags = {}  # dictionary to record potential links

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            # Remove the i-th row and j-th column to compute LSA
            reduced_matrix = np.delete(adj_matrix, i, axis=0)
            reduced_matrix = np.delete(reduced_matrix, j, axis=1)

            input_vector = np.delete(adj_matrix[i], j, axis=0)

            # Apply least-squares approximation
            coeffs = np.linalg.lstsq(reduced_matrix, input_vector, rcond=None)[0]

            feature_matrix = np.delete(adj_matrix, i, axis=0)
            estimated_vector = feature_matrix * coeffs[:, np.newaxis]
            prediction = np.sum(estimated_vector, axis=0)

            # Determine if the predicted value crosses the link threshold
            prediction_flags[(i, j)] = prediction[j] > threshold

    # Create a binary matrix where 1 indicates a predicted missing link
    predicted_matrix = np.zeros_like(adj_matrix)
    for i in range(predicted_matrix.shape[0]):
        for j in range(predicted_matrix.shape[1]):
            if adj_matrix[i, j] == 0 and prediction_flags[(i, j)]:
                predicted_matrix[i][j] = 1

    return predicted_matrix

# Extract the list of predicted links from the matrix
def extract_missing_links(matrix, prediction_matrix, node_list):
    inferred_links = []
    for i in range(prediction_matrix.shape[0]):
        for j in range(prediction_matrix.shape[1]):
            if prediction_matrix[i, j] == 1:
                inferred_links.append((node_list[i], node_list[j]))
    return inferred_links

# Apply the matrix method to our graph
node_list = list(network.nodes())
link_threshold = 0.99
pred_matrix = estimate_links_via_matrix(adj_matrix, link_threshold)
matrix_based_links = extract_missing_links(adj_matrix, pred_matrix, node_list)

print("Missing links identified using Matrix-based method:", len(matrix_based_links))
print(matrix_based_links[8])

# Set similarity threshold for considering a link
jaccard_threshold = 0.134
jaccard_links = []

# Iterate through all possible node pairs
for u in network.nodes():
    for v in network.nodes():
        if u != v and not network.has_edge(u, v):
            neighbors_u = set(network.neighbors(u))
            neighbors_v = set(network.neighbors(v))

            # Calculate Jaccard similarity
            intersection_size = len(neighbors_u & neighbors_v)
            union_size = len(neighbors_u | neighbors_v)

            if union_size > 0:
                similarity = intersection_size / union_size
                if similarity > jaccard_threshold:
                    jaccard_links.append(((u, v), similarity))

print("Missing links identified using Jaccard method:", len(jaccard_links))
print(jaccard_links[8])
