#!/usr/bin/env python3
"""
tetris_random_sequence.py
Generate a random Tetris piece sequence of desired length
Usage:
    python tetris_random_sequence.py 20
"""

import random
import sys

PIECES = ['I','O','T','J','L','S','Z']

def generate_sequence(n):
    return [random.choice(PIECES) for _ in range(n)]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tetris_random_sequence.py <length>")
        sys.exit(1)
    n = int(sys.argv[1])
    seq = generate_sequence(n)
    print("Random Sequence:", seq)
