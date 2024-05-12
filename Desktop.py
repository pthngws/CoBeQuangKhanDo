import math
from simpleai.search import SearchProblem, astar
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
from tkinter import messagebox
import random

# Load images
trap_image = Image.open("wolf.png")
girl_image = Image.open("girl.png")
grandma_image = Image.open("grandma.png")
tree_image = Image.open("tree1.png")
step_image = Image.open("step.png")

# Define cost of moving around the map
cost_regular = 1.0
cost_diagonal = 1.7

# Create the cost dictionary
COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
    "up left": cost_diagonal,
    "up right": cost_diagonal,
    "down left": cost_diagonal,
    "down right": cost_diagonal,
}

# Define the map size
M = 20
N = 20
W = 30

probability_t = 0.3

# Generate the MAP
MAP = [['t' if random.random() < probability_t else ' ' for _ in range(N)] for _ in range(M)]


# Class containing the methods to solve the maze
class MazeSolver(SearchProblem):
    # Initialize the class
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)
        self.is_diagonal_allowed = True  # Default: Diagonal is allowed
        self.explored_cells = set()  # Set to keep track of explored cells

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        super(MazeSolver, self).__init__(initial_state=self.initial)

    def update_costs(self):
        if self.is_diagonal_allowed:
            COSTS.update({
                "up left": cost_diagonal,
                "up right": cost_diagonal,
                "down left": cost_diagonal,
                "down right": cost_diagonal
            })
        else:
            for direction in ["up left", "up right", "down left", "down right"]:
                del COSTS[direction]

    # Define the method that takes actions to arrive at the solution
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if 0 <= newx < N and 0 <= newy < M and self.board[newy][newx] != "#" and self.board[newy][newx] != "t":
                actions.append(action)
                # Tô màu vàng cho ô được xét duyệt
                self.explored_cells.add((newx, newy))
                # Tô màu xanh cho các ô neighbor
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (dx != 0 or dy != 0) and 0 <= newx + dx < N and 0 <= newy + dy < M:
                            self.explored_cells.add((newx + dx, newy + dy))
        return actions

    # Update the state based on the action
    def result(self, state, action):
        x, y = state

        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1

        new_state = (x, y)

        return new_state

    # Check if we have reached the goal
    def is_goal(self, state):
        return state == self.goal

    # Compute the cost of taking an action
    def cost(self, state, action, state2):
        return COSTS[action]

    # Heuristic that we use to arrive at the solution
    def manhattan_distance(self, state):
        x1, y1 = state
        x2, y2 = self.goal
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, state):
        x1, y1 = state
        x2, y2 = self.goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def chebyshev_distance(self, state):
        x1, y1 = state
        x2, y2 = self.goal
        return max(abs(x1 - x2), abs(y1 - y2))

    def diagonal_distance(self, state):
        x1, y1 = state
        x2, y2 = self.goal
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def octile_distance(self, state):
        x1, y1 = state
        x2, y2 = self.goal
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def heuristic(self, state):
        # Choose the desired heuristic here
        return self.manhattan_distance(state)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.dem = 0
        self.so_buoc = 0  # Số bước đã thực hiện
        self.title('Cô bé quàng khăn đỏ')
        self.configure(bg='#E0FFFF')
        self.cvs_me_cung = tk.Canvas(self, width=N*W, height=M*W,
                                      relief=tk.SUNKEN, border=1)
        self.diagonal_allowed = True 

        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.trap_image = ImageTk.PhotoImage(trap_image)
        self.girl_image = ImageTk.PhotoImage(girl_image)
        self.grandma_image = ImageTk.PhotoImage(grandma_image)
        self.tree_image = ImageTk.PhotoImage(tree_image)
        self.step_image = ImageTk.PhotoImage(step_image)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.cvs_me_cung.bind("<Button-1>", self.xu_ly_mouse)
        self.cvs_me_cung.bind("<B1-Motion>", self.xu_ly_mouse)
        self.draw_maze()
        lbl_frm_menu = tk.LabelFrame(self, bg='#FF8247', width=N * W)
        lbl_frm_menu.grid(row=0, column=1, padx=5, pady=7, sticky="NWSE") 
        # Đặt màu nền của LabelFrame thành màu xanh alice

        btn_start = tk.Button(lbl_frm_menu, text='Bắt đầu', width=20, height=2,
                            command=self.btn_start_click, bg='#00FFFF', relief=tk.GROOVE,
                            borderwidth=10, highlightthickness=5)
        btn_reset = tk.Button(lbl_frm_menu, text='Làm mới', width=20, height=2,
                            command=self.btn_reset_click, bg='#7CFC00', relief=tk.GROOVE,
                            borderwidth=10, highlightthickness=5)


        check_diagonal = tk.Checkbutton(lbl_frm_menu, text="Không cho phép đi chéo", command=self.toggle_diagonal, bg='#FF8247')

        # Add radiobuttons for selecting heuristic
        self.selected_heuristic = tk.StringVar(value="Manhattan")
        padx_value = 20  # Giá trị pad-x cố định
        heuristic_label = tk.Label(lbl_frm_menu, text="Chọn heuristic:", bg='#FF8247')
        manhattan_rb = tk.Radiobutton(lbl_frm_menu, text="Manhattan", variable=self.selected_heuristic, value="Manhattan", width=10, padx=padx_value, anchor="w", bg='#FF8247')
        euclidean_rb = tk.Radiobutton(lbl_frm_menu, text="Euclidean", variable=self.selected_heuristic, value="Euclidean", width=10, padx=padx_value, anchor="w", bg='#FF8247')
        chebyshev_rb = tk.Radiobutton(lbl_frm_menu, text="Chebyshev", variable=self.selected_heuristic, value="Chebyshev", width=10, padx=padx_value, anchor="w", bg='#FF8247')
        diagonal_rb = tk.Radiobutton(lbl_frm_menu, text="Diagonal", variable=self.selected_heuristic, value="Diagonal", width=10, padx=padx_value, anchor="w", bg='#FF8247')
        octile_rb = tk.Radiobutton(lbl_frm_menu, text="Octile", variable=self.selected_heuristic, value="Octile", width=10, padx=padx_value, anchor="w", bg='#FF8247')

        # Label to display number of steps
        self.lbl_steps = tk.Label(lbl_frm_menu, text="Số bước: 0", bg='#FF8247')

        btn_start.grid(row=10, column=0, padx=5, pady=5, sticky=tk.SE)  # Thay đổi row thành 8
        btn_reset.grid(row=11, column=0, padx=5, pady=5, sticky=tk.SE)  # Thay đổi row thành 9
        check_diagonal.grid(row=2, column=0, padx=5, pady=5, sticky=tk.N)
        heuristic_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.N)
        # Grid radiobuttons
        manhattan_rb.grid(row=4, column=0, padx=5, pady=5, sticky=tk.N)
        euclidean_rb.grid(row=5, column=0, padx=5, pady=5, sticky=tk.N)
        chebyshev_rb.grid(row=6, column=0, padx=5, pady=5, sticky=tk.N)
        diagonal_rb.grid(row=7, column=0, padx=5, pady=5, sticky=tk.N)
        octile_rb.grid(row=8, column=0, padx=5, pady=5, sticky=tk.N)

        self.lbl_steps.grid(row=12, column=0, padx=5, pady=5, sticky=tk.N)  # Thay đổi row thành 10

        self.cvs_me_cung.grid(row=0, column=0, padx=5, pady=5)


        # Draw the initial maze
  

    def heuristic_selected(self):
        # Mapping between selected heuristic and corresponding method in MazeSolver class
        heuristic_mapping = {
            "Manhattan": "manhattan_distance",
            "Euclidean": "euclidean_distance",
            "Chebyshev": "chebyshev_distance",
            "Diagonal": "diagonal_distance",
            "Octile": "octile_distance"
        }
        return heuristic_mapping[self.selected_heuristic.get()]

    def draw_maze(self):
        # Delete all items on the canvas
        self.cvs_me_cung.delete(tk.ALL)

        # Draw background rectangles to fill the canvas
        for x in range(M):
            for y in range(N):
                self.cvs_me_cung.create_rectangle(x*W, y*W, (x+1)*W, (y+1)*W, fill='#C1FFC1', outline='')

        # Draw images for each cell based on the MAP
        for x in range(M):
            for y in range(N):
                if MAP[y][x] == '#':
                    self.cvs_me_cung.create_image(x*W, y*W, anchor=tk.NW, image=self.trap_image)
                elif MAP[y][x] == 'o':
                    self.cvs_me_cung.create_image(x*W, y*W, anchor=tk.NW, image=self.girl_image)
                elif MAP[y][x] == 'x':
                    self.cvs_me_cung.create_image(x*W, y*W, anchor=tk.NW, image=self.grandma_image)
                elif MAP[y][x] == 't':  # Assume 't' represents a tree
                    self.cvs_me_cung.create_image(x*W, y*W, anchor=tk.NW, image=self.tree_image)

    def xu_ly_mouse(self, event):
        px = event.x
        py = event.y
        x = px // W
        y = py // W
        if self.dem < 2:
            if self.dem == 0:
                MAP[y][x] = 'o'
            elif self.dem == 1:
                MAP[y][x] = 'x'
            self.draw_maze()
            self.dem += 1
        else:
            if MAP[y][x] != 'o' and MAP[y][x] != 'x':
                MAP[y][x] = '#'
                self.draw_maze()

    def btn_start_click(self):
        if self.dem != 2:
            messagebox.showwarning("Warning", "Vui lòng đặt điểm bắt đầu và điểm kết thúc trước khi bắt đầu!")
            return

        # Xóa các đường đi trước đó
        self.cvs_me_cung.delete("path")

        # Đặt lại tổng số bước về 0
        self.so_buoc = 0
        self.lbl_steps.config(text="Số bước: 0")

        problem = MazeSolver(MAP)
        # Set the selected heuristic
        problem.heuristic = getattr(problem, self.heuristic_selected())
        # Run the solver
        result = astar(problem, graph_search=True)

        if result is None:  # Kiểm tra xem thuật toán đã tìm được đường hay không
            messagebox.showinfo("Thông báo", "Không tìm thấy đường đi!")
            return

        # Extract the path
        path = [x[1] for x in result.path()]

        L = len(path)
        for i in range(1, L):
            x = path[i][0]
            y = path[i][1]
            self.cvs_me_cung.create_image(x*W, y*W, anchor=tk.NW, image=self.step_image, tags="path")  # Thêm tags để xóa dễ dàng sau này
            self.cvs_me_cung.update()
            time.sleep(0.1)
            self.so_buoc += 1  # Tăng số bước lên mỗi khi di chuyển
            self.lbl_steps.config(text="Số bước: {}".format(self.so_buoc))  # Cập nhật label hiển thị số bước




    def btn_reset_click(self):
        global MAP
        MAP = [['t' if random.random() < probability_t else ' ' for _ in range(N)] for _ in range(M)]
        self.draw_maze()
        self.dem = 0
        self.so_buoc = 0  # Đặt lại số bước về 0 khi reset
        self.lbl_steps.config(text="Số bước: 0")  # Cập nhật label hiển thị số bước

    def toggle_diagonal(self):
        # Toggle the diagonal movement based on the check button state
        self.diagonal_allowed = not self.diagonal_allowed  # Đảo ngược trạng thái của biến cờ

        problem = MazeSolver(MAP)
        problem.is_diagonal_allowed = self.diagonal_allowed  # Cập nhật giá trị của thuộc tính is_diagonal_allowed
        problem.update_costs()





# Define the maze layout (obstacles)
mau_xanh = np.zeros((W, W, 3), np.uint8) + (np.uint8(96), np.uint8(164), np.uint8(244))
mau_trang = np.zeros((W, W, 3), np.uint8) + (np.uint8(255), np.uint8(255), np.uint8(255))
image = np.ones((M*W, N*W, 3), np.uint8) * 255

for x in range(0, M):
    for y in range(0, N):
        if MAP[x][y] == '#':
            cv2.circle(image, (y*W + W//2, x*W + W//2), W//2, (0, 0, 255), -1)
        elif MAP[x][y] == ' ':
            cv2.rectangle(image, (x*W, y*W), ((x+1)*W, (y+1)*W), (96, 164, 244), -1)  # Xanh alice

color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_converted)

if __name__ == "__main__":
    app = App()
    app.mainloop()