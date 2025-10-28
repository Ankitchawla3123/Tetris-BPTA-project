#!/usr/bin/env python3
"""
tetris_mcts_play_fixed.py

MCTS-driven online Tetris player for a 4x10 board.

Behavior:
 - For each piece in sequence:
    * run MCTS for `mcts_iterations_per_move` iterations from current board
    * select the best action (child with highest average value)
    * play that action (apply gravity + clear lines)
    * print board state and accumulated lines
 - Continue until sequence ends or a piece cannot be placed.

This version fixes the "tuple object has no attribute 'visits'" crash by
ensuring node.children is always a mapping action_key -> child_node.
"""
import numpy as np
import math
import random
import time
import os 
import csv
# ------------------ Piece defs (same as your brute force) ------------------
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
            transformed = [(y, -x) for (x, y) in transformed]
        minx = min(p[0] for p in transformed)
        miny = min(p[1] for p in transformed)
        norm = tuple(sorted(((x-minx, y-miny) for x, y in transformed)))
        variants.add(norm)
    return [list(v) for v in sorted(variants)]

PIECE_ROTATIONS = {k: rotations(v) for k, v in PIECES.items()}

# Board dims
H = 20
W = 10

# ------------------ Board helpers ------------------
def can_place(board, shape, x_off, y_off):
    for sx,sy in shape:
        x,y = sx + x_off, sy + y_off
        if x < 0 or x >= W or y < 0 or y >= H:
            return False
        if board[y, x]:
            return False
    return True

def drop_y_for_x(board, shape, x_off):
    maxy = max(y for x,y in shape)
    spawn_y = H - 1 - maxy
    if not can_place(board, shape, x_off, spawn_y):
        return None
    cur_y = spawn_y
    while cur_y - 1 >= 0 and can_place(board, shape, x_off, cur_y - 1):
        cur_y -= 1
    return cur_y

def place_and_clear(board, shape, x_off, y_off):
    newb = board.copy()
    for sx,sy in shape:
        newb[sy + y_off, sx + x_off] = 1
    # clear full lines
    full = [r for r in range(H) if all(newb[r, :])]
    lines = len(full)
    if lines > 0:
        remain = [newb[r, :] for r in range(H) if r not in full]
        while len(remain) < H:
            remain.append(np.zeros(W, dtype=int))
        newb = np.vstack(remain)
    return newb, lines

def pretty_print_board(board):
    lines = []
    for r in range(H-1, -1, -1):
        lines.append(''.join('#' if c else '.' for c in board[r, :]))
    return '\n'.join(lines)

def legal_moves_for_piece(board, piece):
    moves = []
    if piece not in PIECE_ROTATIONS:
        return moves
    for ridx, shape in enumerate(PIECE_ROTATIONS[piece]):
        max_x = max(x for x,y in shape)
        # same x-off convention as original script
        for x_off in range(0, W - max_x):
            y_drop = drop_y_for_x(board, shape, x_off)
            if y_drop is None:
                continue
            moves.append((ridx, x_off, y_drop, shape))
    return moves

# ------------------ MCTS node & helpers ------------------
class MCTSNode:
    __slots__ = ('board','idx','acc_lines','parent','children','visits','value','untried_actions','action_from_parent')
    def __init__(self, board, idx, acc_lines=0, parent=None, action_from_parent=None):
        self.board = board              # numpy array H x W
        self.idx = idx                  # index in local remaining sequence (0..)
        self.acc_lines = acc_lines      # accumulated lines cleared to reach this node
        self.parent = parent
        # children: dict mapping action_key -> child_node
        # action_key : tuple (ridx, x_off)
        self.children = {}
        self.visits = 0
        self.value = 0.0                # total reward sum (we use total lines as reward)
        self.untried_actions = None     # list of (ridx, x_off, y_drop, shape)
        # action that led from parent to this node (None for root)
        self.action_from_parent = action_from_parent

    def is_terminal(self, sequence):
        return self.idx >= len(sequence)

def uct_score(child, parent_visits, c):
    if child.visits == 0:
        return float('inf')
    exploit = child.value / child.visits
    explore = math.sqrt(math.log(max(1, parent_visits)) / child.visits)
    return exploit + c * explore

def uct_select_child(node, c):
    # returns (action_key, child_node) with highest UCT
    parent_visits = max(1, node.visits)
    best_action = None
    best_child = None
    best_score = -float('inf')
    for action_key, child in node.children.items():
        score = uct_score(child, parent_visits, c)
        if score > best_score:
            best_score = score
            best_action = action_key
            best_child = child
    return best_action, best_child

def default_policy_random_rollout(board, start_idx, acc_lines, sequence, rng):
    cur_board = board.copy()
    total_lines = acc_lines
    for i in range(start_idx, len(sequence)):
        piece = sequence[i]
        moves = legal_moves_for_piece(cur_board, piece)
        if not moves:
            break
        ridx, x_off, y_drop, shape = rng.choice(moves)
        cur_board, lines = place_and_clear(cur_board, shape, x_off, y_drop)
        total_lines += lines
    return total_lines, cur_board

def expand_node(node, sequence, rng):
    # initialize untried actions if necessary
    if node.untried_actions is None:
        if node.idx >= len(sequence):
            node.untried_actions = []
        else:
            piece = sequence[node.idx]
            actions = []
            for ridx, shape in enumerate(PIECE_ROTATIONS[piece]):
                max_x = max(x for x,y in shape)
                for x_off in range(0, W - max_x):
                    y_drop = drop_y_for_x(node.board, shape, x_off)
                    if y_drop is None:
                        continue
                    actions.append((ridx, x_off, y_drop, shape))
            node.untried_actions = actions

    if not node.untried_actions:
        return None

    # pick a random untried action (could be deterministic)
    a = rng.choice(node.untried_actions)
    node.untried_actions.remove(a)
    ridx, x_off, y_drop, shape = a
    new_board, lines = place_and_clear(node.board, shape, x_off, y_drop)
    child = MCTSNode(new_board, node.idx + 1, node.acc_lines + lines, parent=node, action_from_parent=(ridx, x_off))
    action_key = (ridx, x_off)
    node.children[action_key] = child
    return child

def tree_policy(root, sequence, c, rng):
    node = root
    while not node.is_terminal(sequence):
        # initialize untried actions lazily
        if node.untried_actions is None:
            if node.idx >= len(sequence):
                node.untried_actions = []
            else:
                piece = sequence[node.idx]
                actions = []
                for ridx, shape in enumerate(PIECE_ROTATIONS[piece]):
                    max_x = max(x for x,y in shape)
                    for x_off in range(0, W - max_x):
                        y_drop = drop_y_for_x(node.board, shape, x_off)
                        if y_drop is None:
                            continue
                        actions.append((ridx, x_off, y_drop, shape))
                node.untried_actions = actions

        # if there are untried actions, expand one
        if node.untried_actions:
            return expand_node(node, sequence, rng)
        # otherwise select best child by UCT
        if not node.children:
            # dead-end (no moves)
            return node
        _, node = uct_select_child(node, c)
    return node

def backup(node, reward):
    # reward is total lines (not incremental); we add same reward to all visited nodes
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value += reward
        cur = cur.parent

def best_action_from_root(root):
    # choose child with highest average value (value / visits); tie-break by visits
    best_act = None
    best_child = None
    best_avg = -float('inf')
    for act, child in root.children.items():
        if child.visits > 0:
            avg = child.value / child.visits
        else:
            avg = 0.0
        if (avg > best_avg) or (avg == best_avg and child.visits > (best_child.visits if best_child else -1)):
            best_avg = avg
            best_act = act
            best_child = child
    return best_act, best_child

def run_mcts(root_board, sequence, iterations=1000, c=1.4142, seed=None):
    rng = random.Random(seed)
    root = MCTSNode(root_board.copy(), 0, 0, parent=None)
    best_lines_seen = -1
    best_board_seen = None

    for _ in range(iterations):
        node = tree_policy(root, sequence, c, rng)
        if node is None:
            continue
        if node.is_terminal(sequence):
            reward_lines = node.acc_lines
            final_board = node.board
        else:
            reward_lines, final_board = default_policy_random_rollout(node.board, node.idx, node.acc_lines, sequence, rng)
        backup(node, reward_lines)

        if reward_lines > best_lines_seen:
            best_lines_seen = reward_lines
            best_board_seen = final_board.copy()

    return root, best_lines_seen, best_board_seen

# ------------------ Online-play loop ------------------
if __name__ == '__main__':
    # EDIT: sequence to play (allowed letters: I,O,T,S,Z,J,L)
    piece_sequence =['O', 'T', 'T', 'O', 'I', 'I', 'L', 'T', 'J', 'J', 'L', 'T', 'J', 'J', 'I', 'O', 'L', 'I', 'I', 'L', 'O', 'O', 'L', 'I', 'J', 'J', 'I', 'T', 'J', 'I', 'O', 'L', 'T', 'J', 'I', 'I', 'O', 'O', 'J', 'L', 'I', 'O', 'I', 'I', 'O', 'J', 'O', 'J', 'I', 'L']
    # MCTS budget per move
    mcts_iterations_per_move = 2000   # increase for stronger play
    exploration_c = 1.0
    rng_seed = 42

    board = np.zeros((H, W), dtype=int)
    acc_lines = 0

    print("Starting live MCTS play for sequence:", piece_sequence)
    start=time.time()
    for idx in range(len(piece_sequence)):
        piece = piece_sequence[idx]
        remaining_sequence = piece_sequence[idx:]  # include current piece
        print("\n=== Piece #{} -> {} ===".format(idx+1, piece))

        # run MCTS from current state with remaining sequence
        root, best_lines_found, _ = run_mcts(board, remaining_sequence, iterations=mcts_iterations_per_move, c=exploration_c, seed=(rng_seed + idx))

        # choose best action from root according to averaged value
        best = best_action_from_root(root)
        if best[0] is None:
            # no good child found; fallback to deterministic first legal move
            moves = legal_moves_for_piece(board, piece)
            if not moves:
                print("No legal moves available for piece", piece, "-- game over.")
                break
            ridx, x_off, y_drop, shape = moves[0]
            new_board, lines = place_and_clear(board, shape, x_off, y_drop)
            acc_lines += lines
            board = new_board
            print(f"Fallback play: rotation {ridx}, x={x_off}, cleared {lines} lines. Total cleared: {acc_lines}")
            print(pretty_print_board(board))
            continue

        (ridx, x_off), child = best
        # derive chosen shape & drop at current board
        chosen_shape = PIECE_ROTATIONS[piece][ridx]
        y_drop = drop_y_for_x(board, chosen_shape, x_off)
        if y_drop is None:
            # unexpected (MCTS suggested an invalid move), fallback random legal
            moves = legal_moves_for_piece(board, piece)
            if not moves:
                print("No legal moves available for piece", piece, "-- game over.")
                break
            ridx, x_off, y_drop, chosen_shape = random.choice(moves)

        new_board, lines = place_and_clear(board, chosen_shape, x_off, y_drop)
        acc_lines += lines
        board = new_board
        print(f"Chosen play: rotation {ridx}, x={x_off}, cleared {lines} lines. Total cleared: {acc_lines}")
        print(pretty_print_board(board))
        
    
    end=time.time()
    print("\n=== Final result ===")
    print("Total lines cleared:", acc_lines)
    print("Final board:")
    print(pretty_print_board(board))
    
    print("time taken" ,f"{end-start} secs")
    
    results_csv = "tetris_results_MCTS.csv"

    # Check if file already exists
    file_exists = os.path.exists(results_csv)

    # Convert sequence list to string
    sequence_str = ",".join(piece_sequence)

    # Rounded elapsed time
    elapsed_time = round(end - start, 6)

    # Append row
    with open(results_csv, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Add header once
        if not file_exists:
            writer.writerow(["Sequence", "Time (seconds)", "Best Score"])

        writer.writerow([sequence_str, elapsed_time, acc_lines])

    print("✅ CSV updated:", results_csv)

    
