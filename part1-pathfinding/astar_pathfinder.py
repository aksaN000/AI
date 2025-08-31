import heapq as hq

with open('input.txt', 'r') as f:
    graph, h = {}, {}

    for i in f.readlines():
        li = i.strip('\n').split(' ')
        node, h_val = li[0], li[1]
        if int(h_val) == 0:
            g = node
        h[node] = int(h_val)
        child = []
        for j in range(2, len(li), 2):
            child.append((li[j], int(li[j + 1])))
        graph[node] = child

    st = input('Enter start node or press ENTER key to choose default start node: ')
    if st == '':
        st = list(h.keys())[0]
        print('Default start node is', st)


def graph_A_star(st, g, graph, h):
    q = [(h[st], 0, st)]
    parent = {st: None}
    g_cost = {st: 0}
    visited = set()

    while q:
        temp, current_g, current_node = hq.heappop(q)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == g:
            path= []
            while current_node:
                path.append(current_node)
                current_node = parent[current_node]
            return current_g, path[::-1]

        for neighbor, edge_cost in graph[current_node]:
          #  if neighbor in visited:
             #   continue
            tentative_g_cost = current_g + edge_cost
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = g_cost[neighbor] + h[neighbor]
                hq.heappush(q, (f_cost, g_cost[neighbor], neighbor))
                parent[neighbor] = current_node

    return -1, []


cost, path = graph_A_star(st, g, graph, h)


if cost != -1:
    with open('output.txt', "w") as f1:
        f1.write(f"Path: {' -> '.join(path)}\n")
        f1.write(f"Total distance: {cost}\n")
else:
    with open('output.txt', "w") as f1:
        f1.write("No path found.\n")
