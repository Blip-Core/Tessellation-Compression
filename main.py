# TEST SCRIPT FOR CONVERTING ARBITRARY IMAGES INTO TRIANGLES AND LINES
# SCRIPT HAS BEEN CREATED BY @ColeXander ON GITHUB UNDER @Blip-Core
# THIS SCRIPT IS NOT A FINISHED PRODUCT AND IS VERSION 1.0, NOTHING IS FINAL.
# THIS SCRIPT IS PROTECTED UNDER THE GNU GENERAL PUBLIC LICENSE.

from collections import defaultdict
from packaging import version
import importlib.metadata
from pathlib import Path
import operator
import sys

import cv2
import shapely
import numpy as np

REQUIREMENTS = {
    'opencv-python' : '>= 4.0.0.21',
    'shapely' : '>= 2.2.0',
    'numpy' : '>= 2.0.0',
    'Python' : '>= 3.11',
}

COLOR_MAP = [
    # Reds & Deep Oranges
    (0, 0, 255), (0, 35, 255), (0, 70, 255), (0, 105, 255), (0, 140, 255),
    (0, 175, 255), (0, 210, 255), (0, 245, 255), (0, 255, 245), (0, 255, 210),
    # Yellows & Lime Greens
    (0, 255, 175), (0, 255, 140), (0, 255, 105), (0, 255, 70), (0, 255, 35),
    (0, 255, 0), (35, 255, 0), (70, 255, 0), (105, 255, 0), (140, 255, 0),
    # Emerald Greens & Cyans
    (175, 255, 0), (210, 255, 0), (245, 255, 0), (255, 245, 0), (255, 210, 0),
    (255, 175, 0), (255, 140, 0), (255, 105, 0), (255, 70, 0), (255, 35, 0),
    # Cyans & Mid-Blues
    (255, 0, 0), (255, 0, 35), (255, 0, 70), (255, 0, 105), (255, 0, 140),
    (255, 0, 175), (255, 0, 210), (255, 0, 245), (255, 35, 255), (255, 70, 255),
    # Indigos & Deep Purples
    (255, 105, 255), (255, 140, 255), (255, 175, 255), (255, 210, 255), (255, 245, 255),
    (245, 0, 255), (210, 0, 255), (175, 0, 255), (140, 0, 255), (105, 0, 255),
    # Magentas & Hot Pinks
    (70, 0, 255), (35, 0, 255), (70, 70, 255), (100, 100, 255), (130, 130, 255),
    (160, 160, 255), (190, 190, 255), (220, 220, 255), (240, 240, 255), (120, 180, 240),
]

MAX_DISPLAY_DIM = 960
TRIANGLE_BYTES = 7  # 3 vertices (x, y) + 1 color
LINE_BYTES = 6      # 2 endpoints (x, y) + 1 thickness + 1 color

# 4x4 Ordered Bayer matrix normalized to (0, 1)
BAYER_4X4 = (
    np.array(
        [
            [0,  8,  2, 10],
            [12, 4, 14,  6],
            [3, 11,  1,  9],
            [15, 7, 13,  5],
        ],
        dtype=np.float32,
    )
    + 0.5
) / 16.0

# 4x4 Line-Hatch matrix (groups thresholds into horizontal/diagonal runs)
HATCH_4X4 = (
    np.array(
        [
            [0,  1,  2,  3],
            [8,  9, 10, 11],
            [4,  5,  6,  7],
            [12, 13, 14, 15],
        ],
        dtype=np.float32,
    )
    + 0.5
) / 16.0
# 1050% avg

# 4x4 Line-Hatch matrix (groups thresholds into vertical/horizontal runs)
VERT_HATCH_4X4 = (
    np.array(
        [
            [0, 8, 4, 12],
            [1, 9, 5, 13],
            [2, 10, 6, 14],
            [3, 11, 7, 15],
        ],
        dtype=np.float32
    )
    + 0.5
) / 16.0
# 1300% avg

def ensure_requirements_met(requirements: dict) -> tuple[bool, list | None]:
    # Ensures all packages meet their minimum versions. (And also exist)

    class VersionNotMetError(Exception):
        pass

    operators = {
        '>=' : operator.ge,
        '<=' : operator.le,
        '==' : operator.eq,
        '~=' : operator.ne,
    }

    suggested = [f"The following is recommended to fix the issue:"]

    for _, package in enumerate(requirements):
        required = requirements[package]
        
        require_type = required[:2]
        require_version = required[3:]

        if package == 'Python':
            py_version = ".".join(map(str, sys.version_info[:3]))
            py_minor = '.'.join(require_version.split('.')[:2])

            if operators[require_type](version.parse(py_version), version.parse(require_version)):
                continue
            else:
                print(f"This version of Python is not supported!")
                py_version = require_version
                suggested.append("\tPython can be fixed with either of the following:")
                suggested.append(f"\t\tInstall Python with winget: 'winget install -e --id Python.Python.{py_minor} -v {py_version}'")
                suggested.append(f"\t\tInstall the newest version of python at 'https://www.python.org/downloads/release/python-{py_version.replace(".", "")}/'")
                continue

        try:
            current_version = importlib.metadata.version(package)
            
            if operators[require_type](version.parse(current_version), version.parse(require_version)):
                continue

            raise VersionNotMetError((package, current_version, require_version, require_type))
        except importlib.metadata.PackageNotFoundError as missing_package:
            print(f"The package '{package}' could not be found!")
            suggested.append(f'\tInstall {package} using {"'"}python -m pip install "{package}{require_type}{require_version}"{"'"}')
        except VersionNotMetError as outdated_package:
            print(f"The package '{package}' is outdated! ({current_version} ![{require_type}] {require_version})")
            suggested.append(f'\tUpdate {package} using {"'"}python -m pip install --upgrade "{package}{require_type}{require_version}"{"'"}')

    if len(suggested) > 1:
        # Something isnt ready.
        return (False, suggested)
    return (True, None)

def _build_thinning_kernels() -> list[np.ndarray]:
    # Builds 8 Golay L-Kernels for topological hit-or-miss skeletonization.
    # Preserves the 8-connected diagonal dither bridges as continuous lines.
    # https://en.wikipedia.org/wiki/Hit-or-miss_transform

    k1 = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]], dtype=np.int8)
    k2 = np.array([[0, -1, -1], [1, 1, -1], [0, 1, -1]], dtype=np.int8)
    kernels = []
    for r in range(4):
        kernels.append(np.rot90(k1, r))
        kernels.append(np.rot90(k2, r))
    return kernels

THINNING_KERNELS = _build_thinning_kernels()

def resize_for_display(img: np.ndarray, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
    # Locks the largest dimension of the image to max_dim

    h, w = img.shape[:2]
    largest = max(w, h)
    if largest <= 0:
        return img
    scale_factor = max_dim / float(largest)
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def enhance_source_contrast(
    gray_img: np.ndarray,
    clip_percent: tuple[float, float] = (1.0, 99.0),
    clahe_strength: float = 0.30,
) -> np.ndarray:
    # Normalizes the dynamic range via percentile clipping and blends in local contrast enhancement (CLAHE).
    # This results in mid-tone structures standing out cleanly.

    low_p, high_p = np.percentile(gray_img, clip_percent)
    if high_p - low_p < 1e-3:
        stretched = gray_img.copy()
    else:
        stretched = np.clip((gray_img.astype(np.float32) - low_p) * (255.0 / (high_p - low_p)), 0, 255).astype(np.uint8)

    if clahe_strength > 0.0:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
        local_enhanced = clahe.apply(stretched)
        alpha = float(np.clip(clahe_strength, 0.0, 1.0))
        stretched = cv2.addWeighted(stretched, 1.0 - alpha, local_enhanced, alpha, 0)

    return stretched

def _enforce_shade_separation(
    centers: np.ndarray,
    min_gap: float,
    anchor_endpoints: bool = True,
) -> list[int]:
    # Ensure the luminance distance is still at least min_gap for the extracted shades
    # Optionally anchors lightest/darkest shades to 0 and 255

    n = len(centers)
    if n <= 1:
        return [0]

    c = np.sort(centers.astype(np.float64))
    max_possible_gap = 255.0 / (n - 1)
    safe_gap = min(float(min_gap), max_possible_gap * 0.95)

    if anchor_endpoints:
        c[0] = 0.0
        c[-1] = 255.0

    # Iterative forward/backward relaxation to ensure minimum contrast separation
    for _ in range(60):
        for i in range(1, n):
            if c[i] - c[i - 1] < safe_gap:
                c[i] = c[i - 1] + safe_gap
        if anchor_endpoints:
            c[-1] = 255.0
        else:
            c[-1] = min(255.0, c[-1])

        for i in range(n - 2, -1, -1):
            if c[i + 1] - c[i] < safe_gap:
                c[i] = c[i + 1] - safe_gap
        if anchor_endpoints:
            c[0] = 0.0
        else:
            c[0] = max(0.0, c[0])

    return [int(round(val)) for val in c]

def extract_optimal_palette(
    gray_img: np.ndarray,
    n_colors: int = 4,
    min_shade_gap: float = 38.0,
    anchor_endpoints: bool = True,
) -> tuple[list[int], dict[int, int]]:
    # Extracts optimal (contrast aware) grayscale palette of n_colors using
    # Histogram-Weighted 1D K-Means with perceptual edge weighting
    # and minimum gap Repulsion.

    if n_colors < 2:
        raise ValueError("n_colors must be at least 2.")

    # Weight pixels near edges slightly higher so the structual lines
    # influence palette clusters properly
    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.hypot(grad_x, grad_y)
    weights = 1.0 + (edge_mag / (float(np.max(edge_mag)) + 1e-5))

    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256), weights=weights)
    hist = hist.astype(np.float64) + 1e-6 # Avoids empty-bin division by zero
    bins = np.arange(256, dtype=np.float64)

    # Initialize centroids by blending cumulative histogram quantiles
    # (With uniform spacing)
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    quantiles = np.linspace(0.05, 0.95, n_colors)
    init_quantile = np.interp(quantiles, cdf, bins)
    init_uniform = np.linspace(0.0, 255.0, n_colors)
    centers = 0.65 * init_quantile + 0.35 * init_uniform

    # Fast 1D Lloyd's K-Means over the 256 histogram bins
    for _ in range(100):
        prev_centers = centers.copy()
        # Midpoints between adjacent sorted centers help define cluster boundaries
        boundaries = 0.5 * (centers[:-1] + centers[1:])
        cluster_ids = np.searchsorted(boundaries, bins)

        for k in range(n_colors):
            mask = cluster_ids == k
            if np.any(mask):
                centers[k] = np.sum(bins[mask] * hist[mask]) / np.sum(hist[mask])

        # If endpoints anchored to 0 and 255, keep them gently
        # pulled towards extremes
        if anchor_endpoints:
            centers[0] = 0.5 * centers[0]
            centers[-1] = 0.5 * (centers[-1] + 255.0)

        if np.max(np.abs(centers - prev_centers)) < 1e-2:
            break
    
    palette = _enforce_shade_separation(centers, min_gap=min_shade_gap, anchor_endpoints=anchor_endpoints)

    # Build visually distint finish pallete [50..255] for the preview
    finish_levels = np.linspace(50, 255, n_colors).round().astype(int).tolist()
    color_to_finish = dict(zip(palette, finish_levels))

    return palette, color_to_finish

def quantize_with_selective_dither(
    gray_img: np.ndarray,
    palette: list[int],
    match_tolerance: float = 14.0,
    dither_mode: str = "bayer",
    dither_cell_size: int = 1,
) -> tuple[np.ndarray, float]:
    # Quantize grayscale image to extracted `palette`, apply ordered dithering
    # only to pixels in the mid-band between palette shades (proportionally to
    # each interval's shade span).

    pal = np.sort(np.asarray(palette, dtype=np.float32))
    img_f = gray_img.astype(np.float32)
    h, w = img_f.shape

    # Find the bounding palette interval for every pixel
    upper_idx = np.clip(np.searchsorted(pal, img_f, side="right"), 1, len(pal) - 1)
    low_val = pal[upper_idx - 1]
    high_val = pal[upper_idx]
    span = np.maximum(1.0, high_val - low_val)

    # Normalized position t in [0..0, 1.0] between low_val and high_val
    t = (img_f - low_val) / span

    if dither_mode == "none":
        snapped = np.where(t >= 0.5, high_val, low_val).astype(np.uint8)
        return snapped, 0.0
    
    # Scale the deadzone proportionally to reference 85-shade span
    # so narrow & wide intervals get balanced solid-color snapping
    # and smooth mid-interval dithering
    reference_span = 255.0 / max(1, len(pal) - 1)
    tau = float(np.clip(match_tolerance / reference_span, 0.0, 0.49))

    in_dither_zone = (t > tau) & (t < (1.0 - tau))
    dither_pct = float(np.mean(in_dither_zone) * 100.0)

    denom = max(1e-5, 1.0 - 2.0 * tau)
    t_remapped = np.clip((t - tau) / denom, 0.0, 1.0)

    matrix = HATCH_4X4 if dither_mode.lower() == "hatch" else (VERT_HATCH_4X4 if dither_mode.lower() == "vertical hatch" else BAYER_4X4)
    cell = max(1, int(dither_cell_size))
    y_idx = (np.arange(h)[:, None] // cell) % 4
    x_idx = (np.arange(w)[None, :] // cell) % 4
    threshold_map = matrix[y_idx, x_idx]

    quantized = np.where(t_remapped > threshold_map, high_val, low_val).astype(np.uint8)
    return quantized, dither_pct

def compute_static_target(quantized_img: np.ndarray, bg_color: int = 255) -> float:
    # Computes fixed quality target (88% to 98%) from base quantized image.
    foreground_mask = np.uint8(quantized_img != bg_color) * 255
    n_labels, _, _, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
    edge_density = cv2.Laplacian(quantized_img, cv2.CV_64F).var()

    fragmentation_penalty = min(6.0, n_labels / 50.0)
    density_penalty = min(4.0, edge_density / 200.0)
    return max(88.0, 98.0 - fragmentation_penalty - density_penalty)

def get_depth(idx: int, h_matrix: np.ndarray) -> int:
    # Returns tree depth of a contour in a (N, 4) hierarchy matrix.
    depth = 0
    parent = h_matrix[idx, 3]
    while parent != -1:
        depth += 1
        parent = h_matrix[parent, 3]
    return depth

def get_direct_children(idx: int, h_matrix: np.ndarray) -> list[int]:
    # Returns immediate child indices (direct holes) of a contour.
    children = []
    child = h_matrix[idx, 2]
    while child != -1:
        children.append(child)
        child = h_matrix[child, 0]
    return children

def split_thick_and_thin_masks(
    layer_mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Separates a binary layer mask into:
    #     - `thick_mask` : Regions wide enough for clean Delaunay triangulation
    #     - `thin_mask`  : Thin strokes, dither patterns, and fine details better
    #                      represented as lines.

    open_ksize = max(3, int(round(2.0 * scale)) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    thick_mask = cv2.morphologyEx(layer_mask, cv2.MORPH_OPEN, open_kernel)

    diff_mask = cv2.subtract(layer_mask, thick_mask)
    if not np.any(diff_mask):
        return thick_mask, diff_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
    kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
    thick_dilated = cv2.dilate(thick_mask, kernel_3x3, iterations=1)

    touching_ids = np.unique(labels[thick_dilated > 0])
    touches_thick = np.isin(np.arange(num_labels), touching_ids)

    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    max_extent = np.maximum(widths, heights)

    is_corner_burr = touches_thick & (max_extent <= open_ksize)
    is_corner_burr[0] = False

    thick_mask[is_corner_burr[labels]] = 255
    thin_mask = cv2.subtract(layer_mask, thick_mask)

    if np.any(thin_mask):
        thin_mask = cv2.bitwise_and(cv2.dilate(thin_mask, kernel_3x3, iterations=1), layer_mask)
    
    return thick_mask, thin_mask

def skeletonize_thin_mask(thin_mask: np.ndarray, max_iters: int = 12) -> np.ndarray:
    # Thins binary mask of narrow strokes down to 1-pixel-wide topological centerlines.
    skel = thin_mask.copy()
    for _ in range(max_iters):
        prev = skel.copy()
        for kernel in THINNING_KERNELS:
            hit = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel)
            skel = cv2.subtract(skel, hit)
        if np.array_equal(skel, prev):
            break
    return skel

def extract_unique_chains(contour: np.ndarray) -> list[np.ndarray]:
    # Decomposes closed OpenCV skeleton contour into non-overlapping
    # 1-way chains so no segment is drawn or counted twice.

    pts = contour.reshape(-1, 2)
    if len(pts) <= 2:
        return [pts]
    
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    chains: list[np.ndarray] = []
    current_chain = [pts[0]]

    for i in range(len(pts)):
        p1 = (int(pts[i][0]), int(pts[i][1]))
        p2 = (int(pts[(i + 1) % len(pts)][0]), int(pts[(i + 1) % len(pts)][1]))
        if p1 == p2:
            continue

        edge = (p1, p2) if p1 < p2 else (p2, p1)
        if edge not in visited_edges:
            visited_edges.add(edge)
            current_chain.append(pts[(i + 1) % len(pts)])
        else:
            if len(current_chain) >= 2:
                chains.append(np.array(current_chain, dtype=np.int32))
            current_chain = [pts[(i + 1) % len(pts)]]

    if len(current_chain) >= 2:
        chains.append(np.array(current_chain, dtype=np.int32))

    return chains

def vectorize_thin_mask_to_lines(
    thin_mask: np.ndarray,
    width_map: np.ndarray,
    approx_epsilon: float = 0.85,
) -> tuple[dict[int, list[np.ndarray]], int]:
    # Converts binary mask of thin features and dither strokes into simplified polylines
    # grouped by integer stroke thickness for fast batched OpenCV rendering.

    if not np.any(thin_mask):
        return {}, 0

    skel = skeletonize_thin_mask(thin_mask)
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    polyline_buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    segment_count = 0
    h, w = width_map.shape

    for contour in contours:
        pts = contour.reshape(-1, 2)
        n_pts = len(pts)

        if n_pts == 1:
            x, y = int(pts[0, 0]), int(pts[0, 1])
            thickness = max(1, int(round(float(width_map[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]))))
            dot_poly = np.array([[[x, y]], [[x, y]]], dtype=np.int32)
            polyline_buckets[thickness].append(dot_poly)
            segment_count += 1
            continue

        if n_pts == 2:
            xs = np.clip(pts[:, 0], 0, w - 1)
            ys = np.clip(pts[:, 1], 0, h - 1)
            thickness = max(1, int(round(float(np.median(width_map[ys, xs])))))
            polyline_buckets[thickness].append(pts.reshape((-1, 1, 2)).astype(np.int32))
            segment_count += 1
            continue

        for chain in extract_unique_chains(contour):
            xs = np.clip(chain[:, 0], 0, w - 1)
            ys = np.clip(chain[:, 1], 0, h - 1)
            thickness = max(1, int(round(float(np.median(width_map[ys, xs])))))

            simplified = cv2.approxPolyDP(
                chain.reshape((-1, 1, 2)),
                epsilon=approx_epsilon,
                closed=False,
            )
            if len(simplified) >= 2:
                polyline_buckets[thickness].append(simplified)
                segment_count += len(simplified) - 1

    return polyline_buckets, segment_count

def contour_to_valid_polygon(contour: np.ndarray) -> shapely.Polygon | None:
    # Converts OpenCV contour into a valid Shapely polygon if possible.

    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return None
    poly = shapely.Polygon(pts).buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly

def triangulate_shape(
    outer_contour: np.ndarray,
    hole_contours: list[np.ndarray],
    min_ribbon_width: float = 1.25,
) -> list[np.ndarray]:
    # Builds a valid polygon with holes and returns a OpenCV-ready triangle arrays.
    # Returns an empty list if the polygon is too thin so the caller can route it
    # to the line vectorizer instead.

    shell = contour_to_valid_polygon(outer_contour)
    if shell is None:
        return []

    valid_holes = [
        h_poly for h in hole_contours
        if (h_poly := contour_to_valid_polygon(h)) is not None
    ]

    geom = shell.difference(shapely.unary_union(valid_holes)).buffer(0) if valid_holes else shell
    if geom.is_empty or geom.length <= 0:
        return []

    effective_width = (2.0 * geom.area) / geom.length
    if effective_width < min_ribbon_width:
        return []

    triangles_cv = []
    for part in shapely.get_parts(geom):
        if part.geom_type != "Polygon" or part.area <= 0:
            continue
        tri_collection = shapely.constrained_delaunay_triangles(part)
        for tri in shapely.get_parts(tri_collection):
            coords = np.asarray(tri.exterior.coords)[:3]
            if len(coords) == 3:
                triangles_cv.append(np.int32(coords).reshape((-1, 1, 2)))
    
    return triangles_cv

def process_at_scale(
    base_quantized: np.ndarray,
    palette: list[int],
    color_to_finish: dict[int, int],
    scale: float,
    show_live_preview: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    # Upscales selected dithered base image and preforms
    # hybrid thick-triangle + thin-line extraction across
    # the extracted palette.

    base_h, base_w = base_quantized.shape
    width, height = int(round(base_w * scale)), int(round(base_h * scale))

    image = cv2.resize(base_quantized, (width, height), interpolation=cv2.INTER_NEAREST)
    rough_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    bg_color = int(palette[-1])
    fine_image = np.full(image.shape, bg_color, dtype=np.uint8)

    foreground_palette = [c for c in palette if c != bg_color]
    tri_count = 0
    line_count = 0

    for target_level in foreground_palette:
        layer_mask = np.uint8(image == target_level) * 255
        if not np.any(layer_mask):
            continue

        dist_map = cv2.distanceTransform(layer_mask, cv2.DIST_L2, 3)
        local_fg_count = cv2.boxFilter(
            (layer_mask > 0).astype(np.float32),
            ddepth=-1,
            ksize=(3, 3),
            normalize=False,
        )
        width_map = np.maximum(2.0 * dist_map - 1.0, local_fg_count / 3.0)

        # Split layer into thick polygonal regions and thin/dither line regions
        thick_mask, thin_mask = split_thick_and_thin_masks(layer_mask, scale)

        padded_thick = cv2.copyMakeBorder(thick_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, hierarchy = cv2.findContours(
            padded_thick,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_TC89_KCOS,
            offset=(-1, -1),
        )

        finish_level = color_to_finish[target_level]
        finish_green = (0, finish_level, 0)
        finish_line_bgr = (finish_level, finish_level // 2, 255)

        # Triangulate thick contours; route degenerate/ribbon shapes to thin_mask
        if hierarchy is not None:
            h_matrix = hierarchy[0]
            for idx, contour in enumerate(contours):
                depth = get_depth(idx, h_matrix)
                if depth % 2 != 0:
                    continue

                hole_indices = get_direct_children(idx, h_matrix)
                hole_contours = [contours[h_idx] for h_idx in hole_indices]

                mask_color = COLOR_MAP[depth % len(COLOR_MAP)]
                cv2.drawContours(rough_image, contours, idx, mask_color, thickness=-1)
                for h_idx in hole_indices:
                    cv2.drawContours(rough_image, contours, h_idx, (0, 0, 0), thickness=-1)
                
                try:
                    triangles = triangulate_shape(contour, hole_contours)
                except Exception:
                    triangles = []
                
                if triangles:
                    cv2.fillPoly(rough_image, triangles, finish_green)
                    cv2.fillPoly(fine_image, triangles, int(target_level))
                    tri_count += len(triangles)
                else:
                    cv2.drawContours(thin_mask, contours, idx, 255, thickness=-1)
                    for h_idx in hole_indices:
                        cv2.drawContours(thin_mask, contours, h_idx, 0, thickness=-1)

        # Vectorize thin strokes and dither patterns; batch-render by thickness
        polyline_buckets, new_segments = vectorize_thin_mask_to_lines(thin_mask, width_map)
        for thickness, poly_list in polyline_buckets.items():
            cv2.polylines(
                rough_image,
                poly_list,
                isClosed=False,
                color=finish_line_bgr,
                thickness=thickness,
                lineType=cv2.LINE_8,
            )
            cv2.polylines(
                fine_image,
                poly_list,
                isClosed=False,
                color=int(target_level),
                thickness=thickness,
                lineType=cv2.LINE_8,
            )
        line_count += new_segments

        if show_live_preview:
            cv2.imshow("Image processing", resize_for_display(rough_image))
            cv2.imshow("Image Finalization", resize_for_display(fine_image))
            cv2.waitKey(1)

    pixel_diff = cv2.absdiff(image, fine_image)
    mean_error = float(np.mean(pixel_diff))
    true_accuracy = (1.0 - (mean_error / 255.0)) * 100.0

    return fine_image, rough_image, tri_count, line_count, mean_error, true_accuracy

def main(
    image_path: str = "Shade_Test.png",
    n_colors: int = 4,
    auto_contrast: bool = True,
    clip_percent: tuple[float, float] = (1.0, 99.0),
    clahe_strength: float = 0.30,
    min_shade_gap: float = 38.0,
    anchor_endpoints: bool = True,
    initial_scale: float = 2.0,
    scale_step: float = 1.5,
    max_scale: float = 9.0,
    min_improvement: float = 0.05,
    match_tolerance: float = 14.0,
    dither_mode: str = "vertical hatch",
    dither_cell_size: int = 1,
    show_live_preview: bool = True,
) -> None:
    packages_ready, error_message = ensure_requirements_met(REQUIREMENTS)
    if not error_message == None:
        error_message = '\n'.join(error_message)

    if not packages_ready:
        raise ModuleNotFoundError(f"Not all requirements are met.\n{error_message}")
    
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Could not find input image: '{image_path}'.")

    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        raise ValueError(f"Failed to decode image: '{image_path}'.")

    # Pre-enchance global and local contrast before palette extraction
    if auto_contrast:
        base_image = enhance_source_contrast(
            raw_image,
            clip_percent=clip_percent,
            clahe_strength=clahe_strength,
        )
    else:
        base_image = raw_image

    # Automatically extract the optimal contrast-separated palette
    palette, color_to_finish = extract_optimal_palette(
        base_image,
        n_colors=n_colors,
        min_shade_gap=min_shade_gap,
        anchor_endpoints=anchor_endpoints,
    )
    gaps = [palette[i + 1] - palette[i] for i in range(len(palette) - 1)]

    # Quantize with selective dithering tailored to the extracted palette
    base_quantized, dither_pct = quantize_with_selective_dither(
        base_image,
        palette=palette,
        match_tolerance=match_tolerance,
        dither_mode=dither_mode,
        dither_cell_size=dither_cell_size,
    )
    static_target = compute_static_target(base_quantized, bg_color=palette[-1])

    scale = initial_scale
    prev_accuracy: float | None = None
    best_result = None

    while True:
        fine_image, rough_image, tri_count, line_count, mean_error, true_accuracy = process_at_scale(
            base_quantized=base_quantized,
            palette=palette,
            color_to_finish=color_to_finish,
            scale=scale,
            show_live_preview=show_live_preview,
        )

        improvement = 0.0 if prev_accuracy is None else (true_accuracy - prev_accuracy)

        print(f"\n-- Quality Audit (Current scale: {scale:.2f}x) --")
        print(f"Extracted Palette ({n_colors} shades): {palette} (gaps: {gaps})")
        print(f"Selective Dither Coverage:  {dither_pct:.1f}% of pixels (mode='{dither_mode}', tol=±{match_tolerance:.0f})")
        print(f"Mean Pixel Value Deviation: {mean_error:.3f} shades")
        print(f"True Geometric Accuracy:    {true_accuracy:.2f}%")
        print(f"Triangle Count:             {tri_count}")
        print(f"Line Segment Count:         {line_count}")
        if prev_accuracy is not None:
            print(f"Accuracy Improvement:       {improvement:+.2f}%")
        print(f"Target Quality:             {static_target:.2f}%")

        if best_result is None or true_accuracy > best_result["accuracy"]:
            best_result = {
                "scale" : scale,
                "fine_image" : fine_image,
                "rough_image" : rough_image,
                "tri_count" : tri_count,
                "line_count" : line_count,
                "accuracy" : true_accuracy,
            }

        if true_accuracy >= static_target:
            print("-> Target quality achieved.")
            break

        if prev_accuracy is not None and abs(improvement) < min_improvement:
            print(
                f"-> Stopping: accuracy change ({improvement:+.3f}%) is below "
                f"minimum threshold ({min_improvement:.2f}%)."
            )
            break

        if prev_accuracy is not None and improvement < -0.5:
            print(f"-> Stopping: accuracy degraded by {improvement:.2f}%. Reverting to best pass.")
            break

        next_scale = round(scale * scale_step, 2)
        if next_scale > max_scale:
            print(f"-> Stopping: next scale ({next_scale}x) exceeds max_scale ({max_scale}x).")
            break

        print(f"-> Adjusting scale from {scale:.2f}x to {next_scale:.2f}x...")
        prev_accuracy = true_accuracy
        scale = next_scale

    final_fine = best_result["fine_image"]
    final_rough = best_result["rough_image"]
    final_tris = best_result["tri_count"]
    final_lines = best_result["line_count"]

    base_h, base_w = base_image.shape
    original_bytes = base_w * base_h * 4
    vector_bytes = (final_tris * TRIANGLE_BYTES) + (final_lines * LINE_BYTES)

    if vector_bytes > 0:
        compress_ratio = (original_bytes / vector_bytes) * 100.0
        print(
            f"\nCompression Equivalence: {compress_ratio:.0f}% "
            f"({vector_bytes}/{original_bytes} bytes | "
            f"{final_tris} tris, {final_lines} lines at {best_result['scale']:.2f}x scale)"
        )
    else:
        print("\nWarning: No geometric primitives were generated.")

    if show_live_preview:
        cv2.imshow("Image processing", resize_for_display(final_rough))
        cv2.imshow("Image Finalization", resize_for_display(final_fine))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()