#!/usr/bin/env python3
"""
tetris_random_sequence.py
Generate a random Tetris piece sequence of desired length
Usage:
    python tetris_random_sequence.py 20
"""

import random
import sys

PIECES = ['I','O','T','J','L']

def generate_sequence(n):
    return [random.choice(PIECES) for _ in range(n)]

if __name__ == "__main__":
    n = 20
    seq = generate_sequence(50)
    print("Random Sequence:", seq)
