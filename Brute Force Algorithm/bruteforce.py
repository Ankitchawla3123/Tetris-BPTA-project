#!/usr/bin/env python3
"""
tetris_bruteforce_4x10_full.py
Standalone brute-force Tetris placer for a 4x10 board.

Edits: change `piece_sequence` near the bottom to run a custom sequence.
This script explores all placements (rotations + horizontal drops) for the given sequence,
applies gravity and line clears immediately after each placement, and reports the final
boards with the maximum total lines cleared.

WARNING: brute force grows exponentially with sequence length. Keep sequence short (N <= ~7)
if you want reasonable runtime/memory.
"""

import numpy as np
import csv
import time
import os

# --- Piece definitions (each piece as a set of block coords relative to an origin) ---
PIECES = {
    'I': [(0,0),(1,0),(2,0),(3,0)],
    'O': [(0,0),(1,0),(0,1),(1,1)],
    'T': [(0,0),(1,0),(2,0),(1,1)],
    'J': [(0,0),(0,1),(1,0),(2,0)],
    'L': [(0,0),(1,0),(2,0),(2,1)]
}
def rotations(blocks):
    """
    Return unique rotation variants for a piece's blocks.
    Rotation: 90 deg clockwise mapping (x,y) -> (y,-x).
    Each variant is normalized (translated) so min x and min y are 0.
    """
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

# Board dimensions: 4 rows (height) x 10 columns (width)
H = 5
W = 10

def can_place(board, shape, x_off, y_off):
    for sx,sy in shape:
        x,y = sx+x_off, sy+y_off
        if x<0 or x>=W or y<0 or y>=H:
            return False
        if board[y,x]:
            return False
    return True

def drop_y_for_x(board, shape, x_off):
    # Top-most spawn position
    maxy = max(y for x,y in shape)
    spawn_y = H - 1 - maxy

    # If piece cannot spawn at top, return None
    if not can_place(board, shape, x_off, spawn_y):
        return None

    # Drop down as far as possible
    cur_y = spawn_y
    while cur_y-1 >= 0 and can_place(board, shape, x_off, cur_y-1):
        cur_y -= 1
    return cur_y


def place_and_clear(board, shape, x_off, y_off):

    newb = board.copy()
    for sx,sy in shape:
        newb[sy+y_off, sx+x_off] = 1
    # clear full lines
    full = [r for r in range(H) if all(newb[r,:])]
    lines = len(full)
    if lines > 0:
        remain = [newb[r,:] for r in range(H) if r not in full]
        while len(remain) < H:
            remain.append(np.zeros(W,dtype=int))
        newb = np.vstack(remain)
    return newb, lines

def brute_force_all_final_boards(piece_sequence):
    """
    Explore all placement sequences (DFS). Return list of (final_board, total_lines_cleared).
    """
    results = []
    initial = (0, np.zeros((H, W), dtype=int), 0)  # (index, board, accumulated_lines)
    stack = [initial]
    while stack:
        idx, board, acc_lines = stack.pop()
        if idx == len(piece_sequence):
            results.append((board, acc_lines))
            continue
        piece = piece_sequence[idx]
        if piece not in PIECE_ROTATIONS:
            raise ValueError(f"Unknown piece: {piece!r}")
        for shape in PIECE_ROTATIONS[piece]:
            max_x = max(x for x, y in shape)
            # x_off allowed: 0 .. W-1-max_x  (range upper bound is W - max_x)
            for x_off in range(0, W - max_x):
                y_drop = drop_y_for_x(board, shape, x_off)
                if y_drop is None:
                    continue
                new_board, lines_cleared = place_and_clear(board, shape, x_off, y_drop)
                stack.append((idx + 1, new_board, acc_lines + lines_cleared))
    return results

def pretty_print_board(board):
    """Return human-friendly multi-line string of board, top row first."""
    lines = []
    for r in range(H-1, -1, -1):
        lines.append(''.join('#' if c else '.' for c in board[r, :]))
    return '\n'.join(lines)

def save_winners(winners, max_lines, base_name='winners'):
    """
    Save winners to CSV and TXT. CSV rows: 'board_top_to_bottom', 'lines_cleared'
    Board encoding in CSV: 4 rows (top->bottom) joined by '|'.
    TXT contains pretty-printed boards separated by blank lines.
    """
    csv_path = f'{base_name}.csv'
    txt_path = f'{base_name}.txt'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['board_top_to_bottom', 'lines_cleared'])
        for b in winners:
            rows = [''.join('#' if c else '.' for c in b[r, :]) for r in range(H-1, -1, -1)]
            board_str = '|'.join(rows)
            writer.writerow([board_str, max_lines])
    with open(txt_path, 'w') as f:
        f.write(f"Board size: {H}x{W}\n")
        f.write(f"Maximum lines cleared: {max_lines}\n")
        f.write(f"Number of winning boards: {len(winners)}\n\n")
        for i, b in enumerate(winners, start=1):
            f.write(f"--- Winner #{i} ---\n")
            f.write(pretty_print_board(b) + "\n\n")
    return csv_path, txt_path

if __name__ == '__main__':
    # Edit this sequence as needed. Allowed letters: I,O,T,S,Z,J,L
    piece_sequence = ['O','I','O','T','I','L']
    start=time.time()
    print("Running brute-force for sequence:", piece_sequence)
    results = brute_force_all_final_boards(piece_sequence)
    end=time.time()
    print("Explored", len(results), "final states.")
    print("time taken ", f'{end-start} secs')

    if not results:
        print("No valid final states found.")
        raise SystemExit(1)

    max_lines = max(lines for board, lines in results)
    winners = [board for board, lines in results if lines == max_lines]

    print(f"Maximum total lines cleared: {max_lines}. Number of boards achieving it: {len(winners)}")

    csv_path, txt_path = save_winners(winners, max_lines, base_name='winners_L_O_J_I_T_full')
    print("Saved winners CSV:", csv_path)
    print("Saved winners text:", txt_path)

    # Print first 10 winners as a sample
    sample_count = min(10, len(winners))
    print(f"\nSample of first {sample_count} winning boards:")
    for i, b in enumerate(winners[:sample_count], start=1):
        print(f"--- Winner #{i} ---\n{pretty_print_board(b)}\n")
        
    
    results_csv = "tetris_results_bruteforce.csv"

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

        writer.writerow([sequence_str, elapsed_time, max_lines])

    print("✅ CSV updated:", results_csv)
