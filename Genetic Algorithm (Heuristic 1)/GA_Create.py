#!/usr/bin/env python3
"""
tetris_ga_train.py
Genetic Algorithm for Tetris (4x10 board) using 4 weights.
Optimizes weights on the unit 3-sphere.
Real Tetris behavior: pieces cannot spawn if blocked at top.
"""

import numpy as np
import random
import csv

# --- Board size ---
H, W = 20, 10

# --- Tetris pieces ---
PIECES = {
    'I': [(0,0),(1,0),(2,0),(3,0)],
    'O': [(0,0),(1,0),(0,1),(1,1)],
    'T': [(0,0),(1,0),(2,0),(1,1)],
    'J': [(0,0),(0,1),(1,0),(2,0)],
    'L': [(0,0),(1,0),(2,0),(2,1)]
}

def rotations(blocks):
    variants = set()
    for r in range(4):
        transformed = blocks
        for _ in range(r):
            transformed = [(y, -x) for (x, y) in transformed]  # rotate 90°
        minx = min(p[0] for p in transformed)
        miny = min(p[1] for p in transformed)
        norm = tuple(sorted(((x - minx, y - miny) for x, y in transformed)))
        variants.add(norm)
    return [list(v) for v in sorted(variants)]

PIECE_ROTATIONS = {k: rotations(v) for k, v in PIECES.items()}

# --- Board functions ---
def can_place(board, shape, x_off, y_off):
    for sx, sy in shape:
        x, y = sx + x_off, sy + y_off
        if x < 0 or x >= W or y < 0 or y >= H:
            return False
        if board[y, x]:
            return False
    return True

def drop_y_for_x(board, shape, x_off):
    maxy = max(y for x, y in shape)
    spawn_y = H - 1 - maxy

    # --- Real Tetris spawn check ---
    if not can_place(board, shape, x_off, spawn_y):
        return None

    # Drop down as far as possible
    cur_y = spawn_y
    while cur_y - 1 >= 0 and can_place(board, shape, x_off, cur_y - 1):
        cur_y -= 1
    return cur_y

def place_and_clear(board, shape, x_off, y_off):
    newb = board.copy()
    for sx, sy in shape:
        newb[sy + y_off, sx + x_off] = 1
    full = [r for r in range(H) if all(newb[r, :])]
    lines = len(full)
    if lines > 0:
        remain = [newb[r, :] for r in range(H) if r not in full]
        while len(remain) < H:
            remain.append(np.zeros(W, dtype=int))
        newb = np.vstack(remain)
    return newb, lines

# --- Feature extraction ---
def get_features(board, lines_cleared):
    heights = [0] * W
    holes = 0
    for x in range(W):
        col = [board[y][x] for y in range(H)]
        try:
            first = col.index(1)
            heights[x] = H - first
            holes += col[first:].count(0)
        except ValueError:
            heights[x] = 0
    agg_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(W - 1))
    return np.array([agg_height, lines_cleared, holes, bumpiness])

# --- Evaluate genome (lines cleared) ---
def evaluate(weights, n_pieces=30, num_games=100):
    total_lines = 0
    for _ in range(num_games):
        board = np.zeros((H, W), dtype=int)
        lines_cleared_total = 0
        for _ in range(n_pieces):
            piece = random.choice(list(PIECE_ROTATIONS.values()))
            best_val, best_board, best_lines = -1e9, None, 0
            for shape in piece:
                max_x = max(x for x, y in shape)
                for x_off in range(W - max_x):
                    y_drop = drop_y_for_x(board, shape, x_off)
                    if y_drop is None:
                        continue  # skip if cannot spawn
                    new_board, lines = place_and_clear(board, shape, x_off, y_drop)
                    feats = get_features(new_board, lines)
                    val = np.dot(weights, feats)
                    if val > best_val:
                        best_val, best_board, best_lines = val, new_board, lines
            if best_board is None:
                break  # game over if no valid placement
            board = best_board
            lines_cleared_total += best_lines
        total_lines += lines_cleared_total
    return total_lines / num_games  # average lines cleared per game

# --- GA operations ---
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def evolve(pop_size=20, generations=10, offspring_frac=0.3, mutation_prob=0.05):
    # --- Initial population with specific sign pattern ---
    pop = [normalize(np.random.uniform(-1, 1, size=4)) for _ in range(pop_size)]


    with open("ga_weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", "fitness", "w1", "w2", "w3", "w4"])
        for g in range(generations):
            # Evaluate population
            scored = [(evaluate(ind), ind) for ind in pop]
            scored.sort(reverse=True, key=lambda x: x[0])
            print(f"Generation {g}: Best fitness={scored[0][0]:.2f}")
            for fit, ind in scored:
                writer.writerow([g, fit, *ind])

            # --- Create offspring ---
            num_offspring = int(offspring_frac * pop_size)
            offspring = []
            while len(offspring) < num_offspring:
                candidates = random.sample(pop, max(2, pop_size // 10))
                candidate_scores = [(evaluate(c), c) for c in candidates]
                candidate_scores.sort(reverse=True, key=lambda x: x[0])
                (f1, p1), (f2, p2) = candidate_scores[:2]

                # Weighted crossover
                child = normalize(p1 * f1 + p2 * f2)

                # Mutation
                if random.random() < mutation_prob:
                    idx = random.randint(0, 3)
                    child[idx] += random.uniform(-0.2, 0.2)
                    child = normalize(child)

                offspring.append(child)

            # --- Replacement: keep strongest survivors ---
            survivors = [ind for _, ind in scored[:pop_size - num_offspring]]
            pop = survivors + offspring

    # --- Save best genome in NEW file ---
    best_fitness, best_genome = max([(evaluate(ind), ind) for ind in pop], key=lambda x: x[0])
    print("Best genome:", best_genome, "Fitness:", best_fitness)

    # Save to new file
    with open("best_ga_weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fitness", "w1", "w2", "w3", "w4"])
        writer.writerow([best_fitness, *best_genome])

    return best_genome

if __name__ == "__main__":
    best = evolve()
