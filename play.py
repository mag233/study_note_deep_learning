# CNN Convolution Visualizer (single-channel) - Minimal dependencies, heavy comments.
# You can copy this cell into your Jupyter Notebook. It uses only NumPy and Matplotlib.
#
# WHAT THIS CELL DOES
# -------------------
# 1) Implements a simple 2D convolution for a single-channel input (e.g., a grayscale 2D array).
# 2) Draws two figures (no subplots):
#    - Figure A: the padded input array as a heatmap, with a rectangle showing the current k×k window.
#    - Figure B: the full output (feature map) as a heatmap, highlighting the selected output cell.
#
# HOW TO USE
# ----------
# - Change INPUT_H, INPUT_W to set input size.
# - Change STRIDE, PADDING, KERNEL to test different settings.
# - Change OUT_I, OUT_J to select which output cell to visualize.
#
# TERMINOLOGY
# -----------
# - input_matrix:     shape (H, W). The single-channel input.
# - kernel:           shape (k, k). The convolution kernel (weights).
# - stride (S):       how many pixels you move the kernel each step horizontally/vertically.
# - padding (P):      how many zeros to add around the input borders.
# - output (feature map): shape (O, O), where O = (H - k + 2P)/S + 1.
#
# RULES FOLLOWED
# --------------
# - One chart per figure (no subplots).
# - No specific colors are set; Matplotlib chooses defaults.
# - Comments explain parameters and logic in detail.
#
# NOTES
# -----
# This demo targets clarity. It is not vectorized for speed.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def conv2d_single_channel(input_matrix: np.ndarray,
                          kernel: np.ndarray,
                          stride: int = 1,
                          padding: int = 0) -> np.ndarray:
    """
    Perform a valid 2D convolution on a single-channel input with optional zero padding.

    Parameters
    ----------
    input_matrix : np.ndarray
        2D array of shape (H, W). Example: a 6x6 grayscale image.
    kernel : np.ndarray
        2D array of shape (k, k). Example: a 3x3 kernel.
    stride : int, default=1
        Movement step of the kernel. stride=1 visits every adjacent pixel.
        stride=2 skips one pixel between positions, reducing output size.
    padding : int, default=0
        Number of zero rows/cols added to all sides of the input.
        padding=1 adds a 1-pixel border of zeros; padding=0 means no padding.

    Returns
    -------
    output : np.ndarray
        2D array (feature map) with shape:
        O = (H - k + 2*padding) // stride + 1
        so output.shape == (O, O).
    """
    H, W = input_matrix.shape
    k, k2 = kernel.shape
    assert k == k2, "Kernel must be square (k×k)."
    assert stride >= 1, "Stride must be a positive integer."
    assert padding >= 0, "Padding must be a non-negative integer."

    # Zero-pad the input on all sides by 'padding' pixels
    if padding > 0:
        padded = np.pad(input_matrix, ((padding, padding), (padding, padding)),
                        mode='constant', constant_values=0)
    else:
        padded = input_matrix

    Hp, Wp = padded.shape
    O = (Hp - k) // stride + 1
    assert (Hp - k) % stride == 0 and (Wp - k) % stride == 0, (
        "Given H, W, k, padding, and stride, output size is not an integer. "
        "Adjust parameters so (H - k + 2P) and (W - k + 2P) are divisible by stride."
    )

    output = np.zeros((O, O), dtype=float)
    for oi in range(O):          # output row index
        for oj in range(O):      # output col index
            i = oi * stride      # top-left row in padded input for this window
            j = oj * stride      # top-left col in padded input for this window
            window = padded[i:i+k, j:j+k]
            output[oi, oj] = np.sum(window * kernel)
    return output


def draw_grid(ax, H, W):
    """Draw a square grid overlay on current axes for an H×W array visualization."""
    # Vertical lines
    for x in range(W + 1):
        ax.axvline(x - 0.5, linewidth=0.5)
    # Horizontal lines
    for y in range(H + 1):
        ax.axhline(y - 0.5, linewidth=0.5)
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.invert_yaxis()  # put origin at top-left like image arrays
    ax.set_aspect('equal')


def visualize_step(input_matrix: np.ndarray,
                   kernel: np.ndarray,
                   stride: int,
                   padding: int,
                   out_i: int,
                   out_j: int):
    """
    Visualize one convolution step and the corresponding output location.

    Parameters
    ----------
    input_matrix : np.ndarray (H×W)
        Single-channel input.
    kernel : np.ndarray (k×k)
        Convolution kernel.
    stride : int
        Movement step of kernel.
    padding : int
        Zero padding applied equally on all sides.
    out_i : int
        Which output row to visualize (0-based). Change this to move the window vertically.
    out_j : int
        Which output col to visualize (0-based). Change this to move the window horizontally.

    What this function shows
    ------------------------
    Figure A: Padded input with a rectangle overlay indicating the current k×k window.
              The rectangle position is derived from (out_i, out_j, stride).
    Figure B: The entire output (feature map) with the selected output cell framed.
    """
    H, W = input_matrix.shape
    k = kernel.shape[0]

    # Build padded input for display
    if padding > 0:
        padded = np.pad(input_matrix, ((padding, padding), (padding, padding)),
                        mode='constant', constant_values=0)
    else:
        padded = input_matrix
    Hp, Wp = padded.shape

    # Compute the full output first (to check indices and display later)
    output = conv2d_single_channel(input_matrix, kernel, stride=stride, padding=padding)
    O = output.shape[0]

    # Bounds check for requested output coords
    assert 0 <= out_i < O and 0 <= out_j < O, f"out_i/out_j must be in [0, {O-1}]."

    # Map output index -> top-left corner in padded input
    top = out_i * stride
    left = out_j * stride

    # ---- Figure A: Padded input with current window ----
    figA = plt.figure(figsize=(6, 6))
    axA = plt.gca()
    imA = axA.imshow(padded, interpolation='nearest')
    axA.set_title("Padded input and current k×k window\n"
                  f"(k={k}, stride={stride}, padding={padding}, out=({out_i},{out_j}))")
    draw_grid(axA, Hp, Wp)

    # Add a rectangle to show the k×k window. 'edgecolor' defaults are used.
    rect = Rectangle((left-0.5, top-0.5), k, k, fill=False, linewidth=2)
    axA.add_patch(rect)

    # Annotate each cell with its value for clarity (good for small arrays)
    for r in range(Hp):
        for c in range(Wp):
            axA.text(c, r, f"{padded[r, c]:.0f}", ha='center', va='center', fontsize=8)

    # Also annotate the window sum (dot product result) at the rectangle center
    window = padded[top:top+k, left:left+k]
    value = float(np.sum(window * kernel))
    axA.text(left + k/2 - 0.5, top + k/2 - 0.5, f"{value:.1f}",
             ha='center', va='center', fontsize=12, fontweight='bold')

    plt.show()

    # ---- Figure B: Output feature map highlighting the chosen cell ----
    figB = plt.figure(figsize=(6, 6))
    axB = plt.gca()
    imB = axB.imshow(output, interpolation='nearest')
    axB.set_title("Output feature map (highlighted cell)")
    draw_grid(axB, O, O)

    rect_out = Rectangle((out_j-0.5, out_i-0.5), 1, 1, fill=False, linewidth=2)
    axB.add_patch(rect_out)

    # Annotate output numbers
    for r in range(O):
        for c in range(O):
            axB.text(c, r, f"{output[r, c]:.1f}", ha='center', va='center', fontsize=10)

    plt.show()


# =========================
# Demo with tweakable parameters
# =========================

# You can edit these parameters:
INPUT_H, INPUT_W = 6, 6     # input size (H, W)
KERNEL_SIZE = 3              # kernel size k (k×k)
STRIDE = 1                   # stride S
PADDING = 0                  # padding P (0 means 'valid', typical; 1 helps show borders)
OUT_I = 1                    # which output row to visualize (0-based)
OUT_J = 2                    # which output col to visualize (0-based)

# Build a simple input with integers 0..H*W-1 for easy reading
input_matrix = np.arange(INPUT_H * INPUT_W).reshape(INPUT_H, INPUT_W)

# Define a simple kernel. You can modify this to see different effects.
# For example:
# - np.ones((KERNEL_SIZE, KERNEL_SIZE))           -> sums the window
# - np.eye(KERNEL_SIZE)                           -> picks the main diagonal
# - np.flipud(np.fliplr(np.ones((k,k)))) / (k*k)  -> average with flipped orientation
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))

# Visualize one step
visualize_step(input_matrix, kernel,
               stride=STRIDE, padding=PADDING,
               out_i=OUT_I, out_j=OUT_J)
