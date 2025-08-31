import heapq as hq

# Sample graph and heuristic input for testing
graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("D", 2)],
    "C": [("D", 3)],
    "D": []
}
h = {"A": 4, "B": 2, "C": 1, "D": 0}  # Heuristic values
g = "D"  # Goal node

# Implementation of A* algorithm without Check 2 (no early skipping of neighbors)
def graph_A_star_no_check2(st, g, graph, h):
    q = [(h[st], 0, st)]  # Priority queue: (f_cost, g_cost, node)
    parent = {st: None}  # To reconstruct the path
    g_cost = {st: 0}  # To track g_cost of nodes
    visited = set()

    while q:
        _, current_g, current_node = hq.heappop(q)

        if current_node in visited:  # Finalization check
            continue
        visited.add(current_node)

        if current_node == g:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent[current_node]
            return current_g, path[::-1]

        for neighbor, edge_cost in graph[current_node]:
            tentative_g_cost = current_g + edge_cost
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + h[neighbor]
                hq.heappush(q, (f_cost, tentative_g_cost, neighbor))
                parent[neighbor] = current_node

    return -1, []


# Implementation of A* algorithm with properly placed Check 2
def graph_A_star_with_check2(st, g, graph, h):
    q = [(h[st], 0, st)]  # Priority queue: (f_cost, g_cost, node)
    parent = {st: None}  # To reconstruct the path
    g_cost = {st: 0}  # To track g_cost of nodes
    visited = set()

    while q:
        _, current_g, current_node = hq.heappop(q)

        if current_node in visited:  # Finalization check
            continue
        visited.add(current_node)

        if current_node == g:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent[current_node]
            return current_g, path[::-1]

        for neighbor, edge_cost in graph[current_node]:
            tentative_g_cost = current_g + edge_cost
            if neighbor in visited:  # Skip if the neighbor is already finalized
                continue
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + h[neighbor]
                hq.heappush(q, (f_cost, tentative_g_cost, neighbor))
                parent[neighbor] = current_node

    return -1, []


def compare_astar_implementations():
    """Compare the two A* implementations and analyze the differences"""
    start_node = "A"
    
    print("Comparing A* Algorithm Implementations")
    print("=" * 50)
    
    result_no_check2 = graph_A_star_no_check2(start_node, g, graph, h)
    result_with_check2 = graph_A_star_with_check2(start_node, g, graph, h)
    
    print(f"Graph: {graph}")
    print(f"Heuristics: {h}")
    print(f"Start: {start_node}, Goal: {g}")
    print()
    
    print("Results without Check 2 (neighbor visited check):")
    print(f"Cost: {result_no_check2[0]}, Path: {' -> '.join(result_no_check2[1])}")
    print()
    
    print("Results with Check 2 (neighbor visited check):")
    print(f"Cost: {result_with_check2[0]}, Path: {' -> '.join(result_with_check2[1])}")
    print()
    
    print("Analysis:")
    if result_no_check2[0] == result_with_check2[0]:
        print("Both implementations found the same optimal cost.")
    else:
        print("Different costs found - this indicates a potential issue.")
    
    print("\nCheck 2 Impact:")
    print("- Without Check 2: May revisit already processed nodes")
    print("- With Check 2: Skips already finalized neighbors, improving efficiency")
    
    return result_no_check2, result_with_check2


if __name__ == "__main__":
    compare_astar_implementations()
