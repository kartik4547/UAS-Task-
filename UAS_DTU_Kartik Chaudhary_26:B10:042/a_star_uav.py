import heapq
import time
import os
import matplotlib.pyplot as plt

# A* path planning for a small 2D UAV grid
# S = start, G = goal, # = obstacle, . = free space
# The drone can move only up, down, left and right.

moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def heuristic(a, b):
    # Manhattan distance because diagonal movement is not allowed
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_start_goal(grid):
    start = None
    goal = None

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)

    return start, goal


def a_star(grid):
    start, goal = get_start_goal(grid)

    # heap contains (f value, g value, row, column)
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start[0], start[1]))

    # cost from start to every cell we have reached
    g_score = {start: 0}

    # used later to make the path
    parent = {}

    explored = []
    explored_set = set()

    start_time = time.perf_counter()

    while open_list:
        f, g, r, c = heapq.heappop(open_list)
        current = (r, c)

        # If this cell was already expanded, skip it
        if current in explored_set:
            continue

        explored_set.add(current)
        explored.append(current)

        # Goal reached
        if current == goal:
            path = [current]

            while current in parent:
                current = parent[current]
                path.append(current)

            path.reverse()
            run_time = time.perf_counter() - start_time
            return path, explored, len(path) - 1, run_time

        # Check the 4 possible directions
        for dr, dc in moves:
            nr = r + dr
            nc = c + dc

            # outside the grid
            if nr < 0 or nr >= len(grid) or nc < 0 or nc >= len(grid[0]):
                continue

            # obstacle
            if grid[nr][nc] == '#':
                continue

            neighbour = (nr, nc)
            new_g = g + 1

            # Only update if this route is better
            if new_g < g_score.get(neighbour, 999999):
                g_score[neighbour] = new_g
                parent[neighbour] = current
                h = heuristic(neighbour, goal)
                f_new = new_g + h
                heapq.heappush(open_list, (f_new, new_g, nr, nc))

    run_time = time.perf_counter() - start_time
    return None, explored, None, run_time


def draw_grid(grid, path, explored, title, filename):
    # Keep the drawing simple so it is easy to understand
    rows = len(grid)
    cols = len(grid[0])
    picture = [[0 for _ in range(cols)] for _ in range(rows)]

    # 0 free, 1 obstacle, 2 explored, 3 path, 4 start, 5 goal
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '#':
                picture[r][c] = 1

    for cell in explored:
        r, c = cell
        if picture[r][c] == 0:
            picture[r][c] = 2

    if path:
        for r, c in path:
            if grid[r][c] not in ('S', 'G'):
                picture[r][c] = 3

    start, goal = get_start_goal(grid)
    picture[start[0]][start[1]] = 4
    picture[goal[0]][goal[1]] = 5

    plt.figure(figsize=(7, 5))
    plt.imshow(picture, cmap='viridis', vmin=0, vmax=5)
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.grid(True, alpha=0.3)
    plt.title(title)

    plt.text(start[1], start[0], 'S', ha='center', va='center', fontweight='bold')
    plt.text(goal[1], goal[0], 'G', ha='center', va='center', fontweight='bold')

    if path:
        for i, (r, c) in enumerate(path):
            if (r, c) != start and (r, c) != goal:
                plt.text(c, r, str(i), ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# I made a few different maps so the program is not only tested on one grid.
test_cases = [
    ('Test Case 1 - Simple Path', [
        'S.......',
        '........',
        '........',
        '........',
        '........',
        '.......G'
    ]),
    ('Test Case 2 - Multiple Possible Paths', [
        'S........',
        '.#####...',
        '.........',
        '...#####.',
        '.........',
        '...#####.',
        '........G'
    ]),
    ('Test Case 3 - Narrow Passage', [
        'S..#......',
        '.#.#.####.',
        '.#.#......',
        '.#.######.',
        '.#........',
        '.########.',
        '.#.......G',
        '..........'
    ]),
    ('Test Case 4 - Different Obstacle Arrangement', [
        'S.........',
        '..........',
        '......#...',
        '......#...',
        '.#..#..#.#',
        '...#...#..',
        '.........G'
    ]),
    ('Test Case 5 - No Valid Path', [
        'S..#..',
        '.##.##',
        '...#..',
        '##.###',
        '..#..G',
        '######'
    ])
]


def main():
    if not os.path.exists('visualizations'):
        os.mkdir('visualizations')

    log = []
    log.append('UAS-DTU Round 2 - Avionics')
    log.append('Task 1: A* Path Planning')
    log.append('')

    for number, (name, grid) in enumerate(test_cases, start=1):
        path, explored, cost, run_time = a_star(grid)

        image_name = 'visualizations/testcase_%d.png' % number
        draw_grid(grid, path, explored, name, image_name)

        log.append('----------------------------------------')
        log.append(name)
        log.append('Grid:')
        log.extend(grid)

        if path is not None:
            log.append('Path Found: YES')
            log.append('Path:')
            log.append(' -> '.join(str(x) for x in path))
            log.append('Total Path Cost: %d' % cost)
        else:
            log.append('Path Found: NO')
            log.append('No valid path exists.')
            log.append('Total Path Cost: N/A')

        log.append('Nodes Explored: %d' % len(explored))
        log.append('Execution Time: %.8f seconds' % run_time)
        log.append('Visualization: %s' % image_name)
        log.append('')

    with open('simulation_log.txt', 'w') as file:
        file.write('\n'.join(log))

    print('Done. Log saved in simulation_log.txt')


if __name__ == '__main__':
    main()
