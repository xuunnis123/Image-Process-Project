import sys
from PIL import Image

def AStar(start, goal, neighbor_nodes, distance, cost_estimate):
    
    def reconstruct_path(came_from, current_node):
        path = []
        while current_node is not None:
            path.append(current_node)
            current_node = came_from[current_node]
        return list(reversed(path))

    g_score = {start: 0}
    f_score = {start: g_score[start] + cost_estimate(start, goal)}
    openset = {start}
    closedset = set()
    came_from = {start: None}

    while openset:
        current = min(openset, key=lambda x: f_score[x])
        if current == goal:
            return reconstruct_path(came_from, goal)
        openset.remove(current)
        closedset.add(current)
        for neighbor in neighbor_nodes(current):
            if neighbor in closedset:
                continue
            if neighbor not in openset:
                openset.add(neighbor)
            tentative_g_score = g_score[current] + distance(current, neighbor)
            if tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + cost_estimate(neighbor, goal)
    return []

def is_blocked(p):
    x,y = p
    print("x,y=",x,y)
    pixel = path_pixels[x,y]
    if any(c < 225 for c in pixel):
        return True
def von_neumann_neighbors(p):
    x, y = p
    neighbors = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    return [p for p in neighbors if not is_blocked(p)]

def manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def squared_euclidean(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

start = (400, 984)
goal = (398, 25)

# invoke: python mazesolver.py <mazefile> <outputfile>[.jpg|.png|etc.]

path_img = Image.open("maze.jpg")
path_pixels = path_img.load()

distance = manhattan
heuristic = manhattan

path = AStar(start, goal, von_neumann_neighbors, distance, heuristic)

for position in path:
    x,y = position
    path_pixels[x,y] = (255,0,0) # red

path_img.save("maze1.jpg")

'''
from PIL import Image
img = Image.open("/tmp/in.jpg")
(w,h) = img.size
scan = [(394,23)]
while(len(scan) > 0):
    (i,j) = scan.pop()
    (r,g,b) = img.getpixel((i,j))
    if(r*g*b < 9000000):
        img.putpixel((i,j),(210,210,210))
        for x in [i-1,i,i+1]:
            for y in [j-1,j,j+1]:
                scan.append((x,y))
img.save("/tmp/out.png")
'''