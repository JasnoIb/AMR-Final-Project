import numpy as np
import math
import heapq

class Node:
    def __init__(self, x_ind, y_ind, theta_ind, direction, 
                 x, y, theta, steering, cost, parent_index):
        self.x_ind = x_ind
        self.y_ind = y_ind
        self.theta_ind = theta_ind
        self.direction = direction
        self.x = x
        self.y = y
        self.theta = theta
        self.steering = steering
        self.cost = cost
        self.heuristic = 0.0
        self.total_cost = 0.0
        self.parent_index = parent_index

    def __lt__(self, other):
        return self.total_cost < other.total_cost

class HybridAStar:
    def __init__(self, resolution, max_steer):
        self.XY_RES = resolution
        self.YAW_RES = np.radians(15)
        self.MAX_STEER = max_steer
        self.N_STEER = 5
        self.WB = 0.33
        self.MOTION_STEP = resolution * 1.5 
        
        # TUNING PARAMETER: Higher = Robot stays further away
        self.SAFETY_GAIN = 5.0 

    def calc_motion_inputs(self):
        steer_inputs = [0.0]
        for i in range(1, int(self.N_STEER/2) + 1):
            angle = self.MAX_STEER * i / (self.N_STEER / 2)
            steer_inputs.append(angle)
            steer_inputs.append(-angle)
        return steer_inputs

    def move(self, node, length, steering, direction, map_obj):
        x = node.x
        y = node.y
        theta = node.theta

        # Physics Integration
        dist_step = length / 5.0
        for _ in range(5):
            x += direction * dist_step * math.cos(theta)
            y += direction * dist_step * math.sin(theta)
            theta += (direction * dist_step / self.WB) * math.tan(steering)

        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        x_ind = int(round(x / self.XY_RES))
        y_ind = int(round(y / self.XY_RES))
        theta_ind = int(round(theta / self.YAW_RES))

        # Basic movement costs
        steer_cost = 5.0 * abs(steering)
        cost_penalty = 0.0
        if abs(steering - node.steering) > 0.001: cost_penalty += 5.0
        if direction != node.direction: cost_penalty += 50.0
        if direction == -1: cost_penalty += 5.0
        
        proximity_cost = map_obj.get_cost(x, y)
        
        # Add weighted penalty
        safety_penalty = proximity_cost * self.SAFETY_GAIN

        new_cost = node.cost + length + steer_cost + cost_penalty + safety_penalty
        
        return Node(x_ind, y_ind, theta_ind, direction, 
                    x, y, theta, steering, new_cost, None)

    def is_collision(self, node, map_obj):
        return map_obj.is_occupied(node.x, node.y)

    def calc_heuristic(self, node, goal_node):
        return math.hypot(node.x - goal_node.x, node.y - goal_node.y)

    def search(self, start, goal, map_obj):
        sx_ind, sy_ind = map_obj.get_index(start[0], start[1])
        gx_ind, gy_ind = map_obj.get_index(goal[0], goal[1])
        st_ind = int(round(start[2] / self.YAW_RES))
        gt_ind = int(round(goal[2] / self.YAW_RES))

        if map_obj.is_occupied(start[0], start[1]):
            print("Error: Start is in collision!")
            return None
        if map_obj.is_occupied(goal[0], goal[1]):
            print("Error: Goal is in collision!")
            return None

        start_node = Node(sx_ind, sy_ind, st_ind, 1, start[0], start[1], start[2], 0.0, 0.0, None)
        goal_node = Node(gx_ind, gy_ind, gt_ind, 1, goal[0], goal[1], goal[2], 0.0, 0.0, None)

        open_set = {}
        closed_set = {}
        pq = []

        start_node.heuristic = self.calc_heuristic(start_node, goal_node)
        start_node.total_cost = start_node.cost + start_node.heuristic
        
        start_key = (sx_ind, sy_ind, st_ind)
        open_set[start_key] = start_node
        heapq.heappush(pq, start_node)

        steer_inputs = self.calc_motion_inputs()
        directions = [1, -1]
        
        iter_count = 0
        
        while True:
            if not open_set:
                print(f"Failed: Open set empty after {iter_count} iterations.")
                return None

            current = heapq.heappop(pq)
            c_key = (current.x_ind, current.y_ind, current.theta_ind)
            
            if c_key in closed_set: continue
            if c_key in open_set and open_set[c_key].total_cost < current.total_cost: continue

            if c_key in open_set: del open_set[c_key]
            closed_set[c_key] = current

            iter_count += 1
            if iter_count % 2000 == 0:
                print(f"Searching... {iter_count} nodes. Dist: {current.heuristic:.2f}m")

            if (abs(current.x_ind - goal_node.x_ind) <= 2 and 
                abs(current.y_ind - goal_node.y_ind) <= 2):
                
                print(f"Goal Found after {iter_count} steps!")
                path_x, path_y, path_yaw = [], [], []
                temp = current
                while temp is not None:
                    path_x.append(temp.x)
                    path_y.append(temp.y)
                    path_yaw.append(temp.theta)
                    temp = temp.parent_index
                return path_x[::-1], path_y[::-1], path_yaw[::-1]

            for direction in directions:
                for steer in steer_inputs:
                    neighbor = self.move(current, self.MOTION_STEP, steer, direction, map_obj)

                    if self.is_collision(neighbor, map_obj):
                        continue

                    n_key = (neighbor.x_ind, neighbor.y_ind, neighbor.theta_ind)
                    if n_key in closed_set: continue

                    if n_key not in open_set:
                        neighbor.heuristic = self.calc_heuristic(neighbor, goal_node)
                        neighbor.total_cost = neighbor.cost + neighbor.heuristic
                        neighbor.parent_index = current
                        open_set[n_key] = neighbor
                        heapq.heappush(pq, neighbor)
                    else:
                        if open_set[n_key].cost > neighbor.cost:
                            neighbor.heuristic = open_set[n_key].heuristic
                            neighbor.total_cost = neighbor.cost + neighbor.heuristic
                            neighbor.parent_index = current
                            open_set[n_key] = neighbor
                            heapq.heappush(pq, neighbor)