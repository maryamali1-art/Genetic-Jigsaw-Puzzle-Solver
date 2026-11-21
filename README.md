An advanced 2D Jigsaw Puzzle Solver implemented using Genetic Algorithms (GA) and Computer Vision techniques. The system reconstructs shuffled and rotated image pieces by optimizing edge compatibility and minimizing visual discontinuities.

📌 Project Overview

This project applies Genetic Algorithms to solve image reconstruction problems. Given a set of shuffled image pieces, the algorithm attempts to determine their correct positions and rotations to reconstruct the original image.

It combines evolutionary optimization with image processing techniques to improve reconstruction accuracy.

✨ Key Features

Genetic Algorithm: Population initialization, fitness evaluation, selection, crossover, and mutation.

Edge Compatibility: Evaluates the similarity between adjacent puzzle pieces.

LAB Color Space: Uses LAB color representation for color-based edge comparison.

Gradient Prediction: Improves edge matching by considering color gradient continuity.

Edge-Aware Crossover: Preserves cohesive blocks of puzzle pieces during reproduction.

Mutation: Includes piece swapping, rotation, and circular grid shifts.

Elitism & Tournament Selection: Preserves strong solutions and selects promising parents.

Local Search Optimization: Refines the best solution through piece rotations and swaps.

Visualization: Generates reconstruction stages and a fitness convergence curve.

🛠️ Technologies Used

Python

OpenCV

NumPy

Matplotlib

📂 Project Structure

Genetic-Jigsaw-Puzzle-Solver/
│
├── genetic_jigsaw_solver.py
│
├── puzzle_dataset/
│   ├── piece_0.png
│   ├── piece_1.png
│   ├── piece_2.png
│   ├── ...
│   └── metadata.json
│
├── results/
│   ├── reconstructed_puzzle.png
│   ├── evolution_stages.png
│   └── fitness_convergence.png
│
└── README.md

⚙️ Installation

Install the required Python libraries:

pip install numpy opencv-python matplotlib

🚀 How to Run

1. Clone this repository:

git clone https://github.com/maryamali1-art/Genetic-Jigsaw-Puzzle-Solver.git

2. Navigate to the project directory:

cd Genetic-Jigsaw-Puzzle-Solver

3. Place the shuffled puzzle pieces inside the puzzle_dataset directory.

4. Run the main Python script:

python genetic_jigsaw_solver.py

🧩 Dataset Requirements

The system expects individual puzzle pieces in PNG format, following this naming convention:

piece_0.png
piece_1.png
piece_2.png
...

An optional metadata.json file can specify the puzzle grid dimensions:

{
    "grid_rows": 3,
    "grid_cols": 3
}

All puzzle pieces should have consistent dimensions.

🧬 Genetic Algorithm Workflow

The reconstruction process consists of the following stages:

1. Dataset Preparation

Load puzzle pieces and calculate edge compatibility matrices.

2. Population Initialization

Generate random candidate arrangements.

3. Fitness Evaluation

Calculate edge matching errors and evaluate solution quality.

4. Selection

Apply elitism and tournament selection.

5. Crossover

Generate offspring using an edge-aware 2D crossover strategy.

6. Mutation

Apply swapping, rotation, and grid shifting operations.

7. Local Search

Refine the best solution to improve reconstruction quality.

8. Image Reconstruction

Combine the optimized pieces into the final image.

📊 Output Results

After execution, the system generates the following files:

Output

Description

reconstructed_puzzle.png

Final reconstructed image

evolution_stages.png

Visual comparison of reconstruction stages

fitness_convergence.png

Genetic Algorithm fitness convergence curve

🎯 Project Objective

The main objective is to demonstrate how Genetic Algorithms can solve image reconstruction and combinatorial optimization problems by evolving candidate solutions toward better arrangements.

This project also demonstrates the integration of evolutionary computation and digital image processing techniques.

👩‍💻 Author

Developed as an academic project exploring Genetic Algorithms and Digital Image Processing.
