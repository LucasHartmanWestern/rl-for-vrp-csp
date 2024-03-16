import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
import cupy as cp
import warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


# Token (Nx2)
T = np.array([[43.02120034946083, -81.28349087468504],
              [43.004969336049854, -81.18631870502043],
              [42.95923445066671, -81.26016049362336],
              [42.98111190139387, -81.30953935839466],
              [42.9819404397449, -81.2508736429095],
              ])

# Destinations (Mx2)
D = np.array([[42.97520298007788, -81.3206637664334],
              [42.95950149646455, -81.2600673019313],
              [43.01217983061826, -81.27053864527043]
              ])

# Actions (NxMxK)
A = np.array([[[0, 1, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]],
              [[0, 1, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]],
              [[0, 0, 0],
              [1, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 0, 1]],
              [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]],
              [[0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]],
              [[0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]],
              [[0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]],
              ])

# Step size (fixed distance)
step_size = 0.01

def move_tokens_gpu(T, D, A, step_size, iterations):
    # Convert inputs to CuPy arrays
    T = cp.asarray(T)
    D = cp.asarray(D)
    A = cp.asarray(A)

    # Calculate the direction vectors for all iterations
    direction_vectors = (A.reshape((-1, A.shape[-1])) @ D).reshape((iterations, -1, 2)) - T[None, :, :]

    # Normalize the direction vectors in-place
    norms = cp.linalg.norm(direction_vectors, axis=2, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    direction_vectors /= norms

    # Apply the action mask to ensure tokens with [0, 0, 0] actions don't move
    action_mask = cp.any(A.reshape((iterations, -1, A.shape[-1])), axis=2, keepdims=True)
    direction_vectors *= action_mask

    # Calculate the movements for all iterations
    movements = step_size * direction_vectors

    # Cumulative sum of movements to get positions
    positions = cp.cumsum(movements, axis=0) + T[None, :, :]

    # Convert the result back to a NumPy array if needed
    positions = cp.asnumpy(positions)

    return positions

def visualize_movements(T, D, positions):
    fig, ax = plt.subplots()

    # Define a list of colors for the destinations
    destination_colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Plot the destinations as boxes with different colors and add them to the legend
    for i, dest in enumerate(D):
        color = destination_colors[i % len(destination_colors)]  # Cycle through colors if there are more destinations than colors
        rect = patches.Rectangle((dest[0] - 0.001, dest[1] - 0.001), 0.002, 0.002, linewidth=1, edgecolor=color,
                                 facecolor='none', label=f'Destination {i + 1}')
        ax.add_patch(rect)

    # Plot the tokens as connected line segments with dots for each timestep
    for i in range(len(positions[0])):
        x_coords = [pos[i][0] for pos in positions]
        y_coords = [pos[i][1] for pos in positions]
        ax.plot(x_coords, y_coords, marker='o', label=f'Token {i + 1}')

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Token Movements')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':

    start_time = time.time()
    positions = move_tokens_gpu(T, D, A, step_size, A.shape[0])
    end_time = time.time()

    duration = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    print(f"The function took {duration:.10f} milliseconds to run.")

    visualize_movements(T, D, positions)