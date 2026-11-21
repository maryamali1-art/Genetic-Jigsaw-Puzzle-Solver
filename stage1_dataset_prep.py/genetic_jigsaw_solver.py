import os
import glob
import json
import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Genetic Algorithm Hyperparameters
# ==============================================================================
POP_SIZE = 300          # Population size
GENERATIONS = 300       # Number of generations (automatically increases for larger grids like 5x5)
TOURNAMENT_SIZE = 5     # Tournament size for selection
CROSSOVER_RATE = 0.90   # Crossover rate
MUTATION_RATE = 0.45    # Mutation rate
ELITISM_COUNT = 20      # Number of elites

GRID_ROWS = 3
GRID_COLS = 3

# ==============================================================================
# Data Preparation and Edge Gradient Precomputation
# ==============================================================================
def load_dataset(dataset_dir="puzzle_dataset"):
    piece_files = sorted(glob.glob(os.path.join(dataset_dir, "piece_*.png")))
    num_found = len(piece_files)

    if num_found == 0:
        raise FileNotFoundError(f"The directory '{dataset_dir}' is empty or does not contain any piece_*.png images!")

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            grid_r = meta.get("grid_rows", GRID_ROWS)
            grid_c = meta.get("grid_cols", GRID_COLS)
    else:
        side = int(math.isqrt(num_found))
        grid_r, grid_c = (side, side) if side * side == num_found else (GRID_ROWS, GRID_COLS)

    if num_found != grid_r * grid_c:
        raise ValueError(
            f"Data mismatch: found ({num_found}) images, while the grid size is ({grid_r}x{grid_c})!"
        )

    pieces = []
    for fpath in piece_files:
        img = cv2.imread(fpath)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {fpath}")
        pieces.append(img)

    return pieces, grid_r, grid_c

def rotate_piece(img, rot_code):
    if rot_code == 0: return img
    elif rot_code == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot_code == 2: return cv2.rotate(img, cv2.ROTATE_180)
    elif rot_code == 3: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Invalid rotation code")

def precompute_compatibility_matrices(pieces, allowed_rotations):
    """
    Computes compatibility in LAB color space with color gradient continuity prediction
    (Gradient Prediction) to prevent false matches when colors are similar.
    """
    num_p = len(pieces)
    diff_h = np.full((num_p, 4, num_p, 4), np.inf, dtype=np.float64)
    diff_v = np.full((num_p, 4, num_p, 4), np.inf, dtype=np.float64)

    rotated_lab = {}
    for p in range(num_p):
        for r in allowed_rotations:
            rotated_lab[(p, r)] = cv2.cvtColor(rotate_piece(pieces[p], r), cv2.COLOR_BGR2LAB).astype(np.float64)

    for p1 in range(num_p):
        for r1 in allowed_rotations:
            img1 = rotated_lab[(p1, r1)]
            r_edge1 = img1[:, -1, :]
            r_pred1 = 2.0 * img1[:, -1, :] - img1[:, -2, :]
            b_edge1 = img1[-1, :, :]
            b_pred1 = 2.0 * img1[-1, :, :] - img1[-2, :, :]

            for p2 in range(num_p):
                if p1 == p2: continue
                for r2 in allowed_rotations:
                    img2 = rotated_lab[(p2, r2)]
                    l_edge2 = img2[:, 0, :]
                    l_pred2 = 2.0 * img2[:, 0, :] - img2[:, 1, :]
                    t_edge2 = img2[0, :, :]
                    t_pred2 = 2.0 * img2[0, :, :] - img2[1, :, :]

                    diff_h[p1, r1, p2, r2] = np.sum(np.abs(r_edge1 - l_edge2)) + 0.5 * (
                        np.sum(np.abs(r_pred1 - l_edge2)) + np.sum(np.abs(l_pred2 - r_edge1))
                    )
                    diff_v[p1, r1, p2, r2] = np.sum(np.abs(b_edge1 - t_edge2)) + 0.5 * (
                        np.sum(np.abs(b_pred1 - t_edge2)) + np.sum(np.abs(t_pred2 - b_edge1))
                    )

    return diff_h, diff_v

# ==============================================================================
# Step 1: Genetic Representation and Initial Population Initialization
# ==============================================================================
def step1_initialize_population(pop_size, num_pieces, grid_r, grid_c, allowed_rotations):
    population = []
    for _ in range(pop_size):
        piece_indices = list(range(num_pieces))
        random.shuffle(piece_indices)
        
        chromosome = np.empty((grid_r, grid_c), dtype=object)
        idx = 0
        for r in range(grid_r):
            for c in range(grid_c):
                chromosome[r, c] = (piece_indices[idx], random.choice(allowed_rotations))
                idx += 1
        population.append(chromosome)
    return population

# ==============================================================================
# Step 2: Evaluation and Fitness Function
# ==============================================================================
def calculate_edge_error(chromosome, diff_h, diff_v, grid_r, grid_c):
    total_error = 0.0
    for r in range(grid_r):
        for c in range(grid_c - 1):
            p1, rot1 = chromosome[r, c]
            p2, rot2 = chromosome[r, c + 1]
            total_error += diff_h[p1, rot1, p2, rot2]

    for r in range(grid_r - 1):
        for c in range(grid_c):
            p1, rot1 = chromosome[r, c]
            p2, rot2 = chromosome[r + 1, c]
            total_error += diff_v[p1, rot1, p2, rot2]
    return total_error

def step2_evaluate_population(population, diff_h, diff_v, grid_r, grid_c):
    fitnesses = []
    errors = []
    for chromosome in population:
        err = calculate_edge_error(chromosome, diff_h, diff_v, grid_r, grid_c)
        fit = 1_000_000.0 / (1.0 + err)
        errors.append(err)
        fitnesses.append(fit)
    return fitnesses, errors

# ==============================================================================
# Step 3: Selection (Elitism & Tournament)
# ==============================================================================
def step3_select_elites(population, fitnesses, elitism_count):
    sorted_indices = np.argsort(fitnesses)[::-1]
    return [population[sorted_indices[i]].copy() for i in range(elitism_count)]

def step3_tournament_selection(population, fitnesses, k=5):
    competitors = random.sample(range(len(population)), k)
    winner_idx = max(competitors, key=lambda i: fitnesses[i])
    return population[winner_idx]

# ==============================================================================
# Step 4: Edge-Aware 2D Crossover
# ==============================================================================
def step4_crossover_2d(parent1, parent2, diff_h, diff_v, grid_r, grid_c, allowed_rotations):
    """
    Inherits a cohesive block from the first parent, then fills the rest of the grid
    from the second parent, selecting the best rotation angle for the remaining pieces
    based on their neighbors.
    """
    child = np.empty((grid_r, grid_c), dtype=object)
    child.fill(None)
    used_pieces = set()

    # 1. Copy a sub-block from the first parent
    block_h = random.randint(1, max(1, grid_r - 1))
    block_w = random.randint(1, max(1, grid_c - 1))
    start_r = random.randint(0, grid_r - block_h)
    start_c = random.randint(0, grid_c - block_w)

    for r in range(start_r, start_r + block_h):
        for c in range(start_c, start_c + block_w):
            child[r, c] = parent1[r, c]
            used_pieces.add(parent1[r, c][0])

    # 2. Fill empty slots from the second parent
    for r in range(grid_r):
        for c in range(grid_c):
            if child[r, c] is None:
                p_id, rot = parent2[r, c]
                if p_id not in used_pieces:
                    child[r, c] = (p_id, rot)
                    used_pieces.add(p_id)

    # 3. Fill any remaining pieces with the best rotation angle matching existing neighbors
    missing = list(set(range(grid_r * grid_c)) - used_pieces)
    random.shuffle(missing)
    m_idx = 0
    for r in range(grid_r):
        for c in range(grid_c):
            if child[r, c] is None:
                p_id = missing[m_idx]
                m_idx += 1
                best_rot = random.choice(allowed_rotations)
                min_cost = float('inf')
                for rot in allowed_rotations:
                    cost = 0.0
                    if r > 0 and child[r - 1, c] is not None:
                        cost += diff_v[child[r - 1, c][0], child[r - 1, c][1], p_id, rot]
                    if c > 0 and child[r, c - 1] is not None:
                        cost += diff_h[child[r, c - 1][0], child[r, c - 1][1], p_id, rot]
                    if cost < min_cost:
                        min_cost = cost
                        best_rot = rot
                child[r, c] = (p_id, best_rot)

    return child

# ==============================================================================
# Step 5: 2D Genetic Mutation
# ==============================================================================
def step5_mutate_2d(chromosome, grid_r, grid_c, allowed_rotations, mutation_rate):
    mutated = chromosome.copy()

    # Mutation 1: Swap positions of two pieces + probability of rotating one of them
    if random.random() < mutation_rate:
        r1, c1 = random.randint(0, grid_r - 1), random.randint(0, grid_c - 1)
        r2, c2 = random.randint(0, grid_r - 1), random.randint(0, grid_c - 1)
        mutated[r1, c1], mutated[r2, c2] = mutated[r2, c2], mutated[r1, c1]
        if random.random() < 0.5 and len(allowed_rotations) > 1:
            p_id, _ = mutated[r1, c1]
            mutated[r1, c1] = (p_id, random.choice(allowed_rotations))

    # Mutation 2: Rotate a random piece
    if random.random() < mutation_rate and len(allowed_rotations) > 1:
        r, c = random.randint(0, grid_r - 1), random.randint(0, grid_c - 1)
        p_id, current_rot = mutated[r, c]
        other_rots = [rt for rt in allowed_rotations if rt != current_rot]
        mutated[r, c] = (p_id, random.choice(other_rots))

    # Mutation 3: Circular grid shift to fix row or column offsets
    if random.random() < 0.25:
        axis = random.choice([0, 1])
        shift = random.choice([-1, 1])
        mutated = np.roll(mutated, shift=shift, axis=axis)

    return mutated

# ==============================================================================
# Step 6: Joint Swap + Rotation Local Search
# ==============================================================================
def step6_local_search(chromosome, diff_h, diff_v, grid_r, grid_c, allowed_rotations):
    """
    Tests swapping any two pieces while trying all possible rotation angles for both together,
    in addition to circular grid shifts to break out of any local optima.
    """
    current = chromosome.copy()
    best_err = calculate_edge_error(current, diff_h, diff_v, grid_r, grid_c)
    improved = True

    while improved:
        improved = False

        # 1. Test rotating each piece individually
        for r in range(grid_r):
            for c in range(grid_c):
                p_id, orig_rot = current[r, c]
                for rot in allowed_rotations:
                    if rot == orig_rot: continue
                    candidate = current.copy()
                    candidate[r, c] = (p_id, rot)
                    err = calculate_edge_error(candidate, diff_h, diff_v, grid_r, grid_c)
                    if err < best_err:
                        best_err = err
                        current = candidate
                        improved = True

        # 2. Test swapping any two pieces while trying all rotation angles for both together
        for r1 in range(grid_r):
            for c1 in range(grid_c):
                for r2 in range(grid_r):
                    for c2 in range(grid_c):
                        if (r1, c1) >= (r2, c2): continue
                        p1, _ = current[r1, c1]
                        p2, _ = current[r2, c2]
                        for rot1 in allowed_rotations:
                            for rot2 in allowed_rotations:
                                candidate = current.copy()
                                candidate[r1, c1] = (p2, rot2)
                                candidate[r2, c2] = (p1, rot1)
                                err = calculate_edge_error(candidate, diff_h, diff_v, grid_r, grid_c)
                                if err < best_err:
                                    best_err = err
                                    current = candidate
                                    improved = True

        # 3. Test circular grid shift
        for sr in range(grid_r):
            for sc in range(grid_c):
                if sr == 0 and sc == 0: continue
                candidate = np.roll(current, shift=(sr, sc), axis=(0, 1))
                err = calculate_edge_error(candidate, diff_h, diff_v, grid_r, grid_c)
                if err < best_err:
                    best_err = err
                    current = candidate
                    improved = True

    return current, best_err

# ==============================================================================
# Reconstruction and Plotting Functions
# ==============================================================================
def reconstruct_image(chromosome, pieces, grid_r, grid_c):
    rows_img = []
    for r in range(grid_r):
        row_pieces = [rotate_piece(pieces[chromosome[r, c][0]], chromosome[r, c][1]) for c in range(grid_c)]
        rows_img.append(np.hstack(row_pieces))
    return np.vstack(rows_img)

def save_evolution_snapshots(snapshots, output_path="results/evolution_stages.png"):
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, (title, img_bgr) in zip(axes, snapshots):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ==============================================================================
# Main Genetic Algorithm Engine (Main GA Loop)
# ==============================================================================
def run_genetic_algorithm():
    print("=" * 65)
    print("  🧬 Advanced Genetic Algorithm Puzzle Solver (2D GA + Gradient)")
    print("=" * 65)

    pieces, grid_r, grid_c = load_dataset()
    num_pieces = len(pieces)
    h, w = pieces[0].shape[:2]
    allowed_rotations = [0, 1, 2, 3] if h == w else [0, 2]

    # Automatically scale up resources when working on a 5x5 grid (25 pieces)
    active_pop_size = 500 if num_pieces > 9 else POP_SIZE
    active_generations = 700 if num_pieces > 9 else GENERATIONS

    print(f"[Preparation] Loaded {num_pieces} pieces ({grid_r}x{grid_c}) | Allowed rotations: {len(allowed_rotations)}")
    diff_h, diff_v = precompute_compatibility_matrices(pieces, allowed_rotations)

    # --- Step 1: Population Initialization ---
    print(f"\n[Step 1: Initialization] Creating a random population of {active_pop_size} chromosomes...")
    population = step1_initialize_population(active_pop_size, num_pieces, grid_r, grid_c, allowed_rotations)

    best_chromosome = None
    best_fitness = -1.0
    best_error = float('inf')
    fitness_history = []
    snapshots = []
    stagnation_counter = 0

    snapshot_gens = [1, max(2, active_generations // 12), max(5, active_generations // 3)]

    # --- Start Evolution Loop Across Generations ---
    print(f"[Steps 2-5: Evolution] Starting evaluation, selection, crossover, and mutation over {active_generations} generations:\n")
    for gen in range(1, active_generations + 1):
        
        # Step 2: Evaluation
        fitnesses, errors = step2_evaluate_population(population, diff_h, diff_v, grid_r, grid_c)
        
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_error = errors[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_history.append(best_fitness)

        if gen in snapshot_gens:
            snapshots.append((f"Generation {gen}\n(Error: {best_error:.0f})", reconstruct_image(best_chromosome, pieces, grid_r, grid_c)))

        if gen % 20 == 0 or gen == 1:
            print(f"  ► Gen {gen:03d} | Best Fitness: {best_fitness:8.2f} | Min Edge Error: {best_error:8.1f}")

        # Handle genetic stagnation
        if stagnation_counter > 40:
            elites = step3_select_elites(population, fitnesses, ELITISM_COUNT)
            fresh_blood = step1_initialize_population(active_pop_size - ELITISM_COUNT, num_pieces, grid_r, grid_c, allowed_rotations)
            population = elites + fresh_blood
            stagnation_counter = 0
            continue

        # Step 3: Elitism
        next_generation = step3_select_elites(population, fitnesses, ELITISM_COUNT)

        # Steps 3, 4 & 5: Selection -> Crossover -> Mutation
        while len(next_generation) < active_pop_size:
            parent1 = step3_tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE)
            parent2 = step3_tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_RATE:
                child = step4_crossover_2d(parent1, parent2, diff_h, diff_v, grid_r, grid_c, allowed_rotations)
            else:
                child = parent1.copy()

            child = step5_mutate_2d(child, grid_r, grid_c, allowed_rotations, MUTATION_RATE)
            next_generation.append(child)

        population = next_generation

    # --- Step 6: Joint Local Search ---
    print("\n[Step 6: Joint Local Search] Refining the final result to achieve the most accurate match...")
    final_chromosome, final_error = step6_local_search(best_chromosome, diff_h, diff_v, grid_r, grid_c, allowed_rotations)
    print(f"  ✔ Final edge error after refinement: {final_error:.1f}")

    os.makedirs("results", exist_ok=True)
    final_img = reconstruct_image(final_chromosome, pieces, grid_r, grid_c)
    cv2.imwrite("results/reconstructed_puzzle.png", final_img)

    snapshots.append((f"Final Result\n(Error: {final_error:.0f})", final_img))
    save_evolution_snapshots(snapshots, "results/evolution_stages.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, color='#8e44ad', linewidth=2.5, label="Best Fitness")
    plt.title("Genetic Algorithm Convergence Curve", fontsize=13, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("results/fitness_convergence.png", dpi=150)
    plt.close()

    print("\n" + "=" * 65)
    print("  ✅ All Genetic Algorithm steps completed successfully!")
    print("  📁 Final Image: results/reconstructed_puzzle.png")
    print("  📁 Visual Evolution Stages: results/evolution_stages.png")
    print("  📁 Convergence Curve: results/fitness_convergence.png")
    print("=" * 65)

if __name__ == "__main__":
    run_genetic_algorithm()