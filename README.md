# 🧩 Tetris AI Algorithms

This repository explores **different AI approaches** to play the classic **Tetris game**, comparing their performance and behavior using various heuristic and search-based strategies.

---

## 📁 Project Structure

```
Tetris-AI/
│
├── Brute Force Algorithm/
│   └── Implementation of a complete search-based Tetris player
│
├── Genetic Algorithm (Heuristic 1)/
│   └── Evolutionary Tetris player using GA-optimized heuristics
│
├── MCTS ALGO (Heuristic 2)/
│   └── Monte Carlo Tree Search–based Tetris AI
│
└── tetris_random_sequence.py
    └── Generates a random sequence of Tetris pieces (for consistent testing)
```

---

## 🧠 Overview of Algorithms

### 1️⃣ Brute Force Algorithm
- Explores **all possible placements** for the current piece and future lookahead (depth-based search).
- Selects the **best-scoring board state** using handcrafted heuristic evaluation.
- Acts as a baseline for comparison.

**Key Idea:** Exhaustive search to find the best move, limited by computational time.

---

### 2️⃣ Genetic Algorithm (Heuristic 1)
- Uses **evolutionary computation** to learn optimal **heuristic weights**.
- Each genome (set of weights) evaluates board features such as:
  - Aggregate height  
  - Lines cleared  
  - Holes  
  - Bumpiness  
- The algorithm evolves the population to maximize lines cleared over generations.

**Key Idea:** Learn a weighted evaluation function using GA.

---

### 3️⃣ MCTS Algorithm (Heuristic 2)
- Implements **Monte Carlo Tree Search (MCTS)** for decision-making in Tetris.
- Combines:
  - Random simulations (rollouts)
  - Tree expansion and backpropagation
- Uses **Upper Confidence Bound (UCT)** to balance exploration and exploitation.

**Key Idea:** Look ahead by simulating possible future piece sequences without exhaustive search.

---

### 4️⃣ `tetris_random_sequence.py`
- Generates deterministic random sequences of Tetris pieces.
- Used across all algorithms for **fair and consistent testing**.

---

