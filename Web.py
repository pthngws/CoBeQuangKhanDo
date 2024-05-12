import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import heapq
import math

# Define the map size
M = 20
N = 20

# Load the tree image
# Load the tree image
tree_image = Image.open("CoBeQuangKhanDo/Image/tree.png")
girl_image = Image.open("CoBeQuangKhanDo/Image/girl.png")
grandma_image = Image.open("CoBeQuangKhanDo/Image/grandma.png")
step_image = Image.open("CoBeQuangKhanDo/Image/step.png")
wolf_image = Image.open("CoBeQuangKhanDo/Image/wolf.png")


# Define global variables for the plot
fig, ax = plt.subplots(figsize=(5, 5))


# Function to draw the map with start, end, and path
def draw_map(map_data, start, end, path=None):
    ax.imshow(np.zeros((M, N, 3)) + [0.7, 0.9, 0.8])  # Background color
    for x in range(M):
        for y in range(N):
            if map_data[x][y] == 't':
                ax.imshow(tree_image, extent=(y - 0.5, y + 0.5, x - 0.5, x + 0.5), origin='lower')  # Tree
            if map_data[x][y] == '#':
                ax.imshow(wolf_image, extent=(y - 0.5, y + 0.5, x - 0.5, x + 0.5), origin='lower')  # Tree
    ax.imshow(girl_image, extent=(start[1] - 0.5, start[1] + 0.5, start[0] - 0.5, start[0] + 0.5), origin='lower')  # Start point
    ax.imshow(grandma_image, extent=(end[1] - 0.5, end[1] + 0.5, end[0] - 0.5, end[0] + 0.5), origin='lower')  # End point

    # Draw path if available
    if path:
        for node in path[1:-1]:  # Skip the first and last nodes
            ax.imshow(step_image, extent=(node[1] - 0.5, node[1] + 0.5, node[0] - 0.5, node[0] + 0.5), origin='lower')

    ax.invert_yaxis()  # Invert y-axis to have (0, 0) at the top-left
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))

    st.pyplot(fig)


# Define the heuristic function (Manhattan distance)
def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define the heuristic function (Euclidean distance)
def heuristic_euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Define the heuristic function (Chebyshev distance)
def heuristic_chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Define the heuristic function (Diagonal distance)
def heuristic_diagonal(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# Define the heuristic function (Octile distance)
def heuristic_octile(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# A* algorithm to find the path
def astar_search(map_data, start, end, di_chuyen_cheo, heuristic_type):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, (0, start, []))

    while open_list:
        _, current, path = heapq.heappop(open_list)

        if current == end:
            return path + [current]

        if current in closed_set:
            continue

        closed_set.add(current)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if di_chuyen_cheo:
            directions.extend([(1, 1), (-1, 1), (1, -1), (-1, -1)])

        for dx, dy in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < M and 0 <= neighbor[1] < N and map_data[neighbor[0]][neighbor[1]] != 't'and map_data[neighbor[0]][neighbor[1]] != '#':
                if heuristic_type == "Manhattan":
                    h = heuristic_manhattan(neighbor, end)
                elif heuristic_type == "Euclidean":
                    h = heuristic_euclidean(neighbor, end)
                elif heuristic_type == "Chebyshev":
                    h = heuristic_chebyshev(neighbor, end)
                elif heuristic_type == "Diagonal":
                    h = heuristic_diagonal(neighbor, end)
                elif heuristic_type == "Octile":
                    h = heuristic_octile(neighbor, end)
                else:
                    raise ValueError("Invalid heuristic type")

                new_cost = len(path) + 1 + h
                heapq.heappush(open_list, (new_cost, neighbor, path + [current]))

    return None


# Update the main function to include A* path finding
def main():
    st.title("Cô bé quàng khăn đỏ")
    MAP = [
        [' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', 't', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', 't', ' ', 't', 't', ' '],
        [' ', ' ', ' ', 't', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', 't', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 't', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' '],
        [' ', ' ', 't', ' ', ' ', ' ', '#', ' ', ' ', ' ', 't', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 't', 't', ' ', ' ', ' ', 't', 't', ' ', ' ', ' ', ' ', 't', ' ', ' '],
        [' ', 't', ' ', ' ', 't', ' ', ' ', 't', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', 't', ' ', ' ', ' ', ' ', '#', 't', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', 't', ' ', ' ', ' ', ' ', ' ', 't', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', 't', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', 't', '#', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', 't', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', 't', ' ', ' ', ' ', ' ', ' ', ' '],
    ]

    start_coords = st.text_input("Cô bé quàng khăn đỏ (x, y):", "0, 0")
    end_coords = st.text_input("Bà ngoại (x, y):", f"{N-1}, {M-1}")

    start_x, start_y = map(int, start_coords.split(','))
    end_x, end_y = map(int, end_coords.split(','))

    di_chuyen_cheo = st.checkbox("Cho phép di chuyển chéo")

    heuristic_type = st.radio("Chọn heuristic:", ["Manhattan", "Euclidean", "Chebyshev", "Diagonal", "Octile"])

    start = (start_y, start_x)
    end = (end_y, end_x)

    path = astar_search(MAP, start, end, di_chuyen_cheo, heuristic_type)

    if path:
        draw_map(MAP, start, end, path=path)
    else:
        st.write("Không tìm thấy đường đi.")

if __name__ == "__main__":
    main()

