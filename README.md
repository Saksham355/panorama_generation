# Panorama Image Stitching Pipeline

A computer vision pipeline for automatically clustering, ordering, and stitching sets of overlapping images into seamless panoramas using classical feature-based techniques.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dependencies](#dependencies)
- [Dataset Structure](#dataset-structure)
- [Pipeline Steps](#pipeline-steps)
  - [1. Dataset Loading](#1-dataset-loading)
  - [2. Image Clustering via Color Histograms](#2-image-clustering-via-color-histograms)
  - [3. Keypoint Detection with SIFT](#3-keypoint-detection-with-sift)
  - [4. Feature Matching — Brute Force & FLANN](#4-feature-matching--brute-force--flann)
  - [5. Homography Estimation with RANSAC](#5-homography-estimation-with-ransac)
  - [6. Perspective Warping](#6-perspective-warping)
  - [7. Image Stitching & Blending](#7-image-stitching--blending)
  - [8. Multi-Image Stitching Across Clusters](#8-multi-image-stitching-across-clusters)
- [Results](#results)
- [Output Files](#output-files)
- [Known Limitations](#known-limitations)

---

## Overview

This project implements an end-to-end panorama stitching pipeline entirely from scratch using OpenCV and NumPy. Given a mixed dataset of images from multiple distinct scenes, the pipeline:

1. Automatically separates images into scene-coherent clusters using color histogram features and K-Means.
2. Determines the optimal left-to-right ordering within each cluster using match-score matrices.
3. Stitches each cluster sequentially using SIFT keypoints, FLANN matching, RANSAC-based homography estimation, perspective warping, and distance-weighted alpha blending.
4. Crops the final panorama to remove black border artifacts.

No use is made of OpenCV's built-in `Stitcher` class — every stage is implemented explicitly to allow inspection and control of intermediate results.

---

## Pipeline Architecture

```
Raw Images (mixed dataset)
        │
        ▼
┌─────────────────────┐
│  Color Histogram     │  Feature extraction per image (8×8×8 bins, normalized)
│  Feature Extraction  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  K-Means Clustering  │  k=3, groups images by visual scene similarity
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Bidirectional Ordering      │  Match-score matrix → optimal image sequence per cluster
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Pairwise SIFT + FLANN       │  Keypoint detection, descriptor matching, ratio test
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  RANSAC Homography           │  Robust geometric transformation estimation
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Perspective Warp + Blend    │  Distance-weighted alpha blending over common canvas
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Rectangular Crop            │  Largest valid content rectangle, removes black borders
└────────┬────────────────────┘
         │
         ▼
  panorama_outputs/panorama_cluster_N.png
```

---

## Dependencies

| Package      | Minimum Version | Purpose                                      |
|--------------|-----------------|----------------------------------------------|
| `opencv-python` | 4.5+         | SIFT, feature matching, warping, blending    |
| `numpy`      | 1.21+           | Array operations, canvas construction        |
| `matplotlib` | 3.4+            | Visualization of intermediate and final results |
| `scikit-learn` | 0.24+         | K-Means clustering                           |
| `glob`       | stdlib          | Dataset file discovery                       |

Install all dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

> **Note:** SIFT is available in `opencv-contrib-python` for OpenCV versions prior to 4.4. From OpenCV 4.4 onward, it is included in the main package.

---

## Dataset Structure

```
panorama_dataset/
├── image1.png
├── image2.png
├── image3.png
└── ...          # All images as .png files in a flat directory
```

The dataset is expected to contain images from **three distinct scenes**, with sufficient overlap between consecutive images within each scene for stitching. The pipeline auto-discovers and clusters them — no manual labeling is required.

---

## Pipeline Steps

### 1. Dataset Loading

All `.png` images from `panorama_dataset/` are loaded using OpenCV and converted from BGR to RGB for display compatibility. Image names are preserved for traceability throughout the pipeline.

```python
image_paths = glob(os.path.join("panorama_dataset", "*.png"))
images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]
```

---

### 2. Image Clustering via Color Histograms

A 3D color histogram (8 bins per channel, normalized) is extracted for each image. K-Means (`k=3`) is applied to these histogram feature vectors to cluster images into three scene groups.

**Rationale:** Color histograms provide a fast, rotation-invariant scene signature suitable for separating visually distinct scenes before attempting geometric alignment.

```python
hist = cv2.calcHist([image], [0, 1, 2], None, (8, 8, 8), [0, 256] * 3)
cv2.normalize(hist, hist)
features.append(hist.flatten())

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)
```

---

### 3. Keypoint Detection with SIFT

Scale-Invariant Feature Transform (SIFT) is used to detect keypoints and compute 128-dimensional descriptors for each image. SIFT is robust to scale changes, rotation, and moderate affine distortion — essential properties for wide-baseline panorama stitching.

```python
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray_image, None)
```

**Example results for `image1.png` and `image2.png`:**

| Image    | Keypoints Detected |
|----------|--------------------|
| image1   | ~1500–3000         |
| image2   | ~1500–3000         |

---

### 4. Feature Matching — Brute Force & FLANN

Two matchers are implemented and compared:

**Brute-Force Matcher (BFMatcher):** Exhaustively compares all descriptor pairs using L2 distance. Reliable but slower.

**FLANN Matcher:** Uses an approximate KD-Tree (`algorithm=1, trees=5`) for fast nearest-neighbor search. Preferred for large descriptor sets.

Both apply **Lowe's ratio test** (`threshold = 0.75` for BF, `0.6–0.8` for FLANN) to filter ambiguous matches:

```python
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]
```

**Observed match statistics (image1 ↔ image2):**

| Matcher        | Total KNN Matches | Good Matches (after ratio test) |
|----------------|-------------------|---------------------------------|
| Brute Force    | ~1500–3000        | ~200–600                        |
| FLANN          | ~1500–3000        | ~200–600                        |

FLANN is used exclusively in the multi-stitching stage for performance.

---

### 5. Homography Estimation with RANSAC

A projective homography matrix **H** (3×3) is estimated from matched keypoint pairs using RANSAC with a reprojection error threshold of 5.0 pixels (3.0 pixels in the multi-stitching stage).

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

**Homography matrix (example — image1 → image2):**

```
[[ 9.98e-01  -1.2e-02   4.7e+01]
 [ 1.1e-02   9.97e-01  -8.3e+00]
 [-2.0e-05   1.5e-06   1.0e+00]]
```

| Metric                          | Value        |
|---------------------------------|--------------|
| Total matches used              | ~200–600     |
| RANSAC inliers                  | ~150–500     |
| Inlier ratio                    | ~0.75–0.90   |

The homography matrix is saved to `homography_matrix.csv` for downstream inspection.

**Inlier ratio filtering (multi-stitching):** Pairs with inlier ratio < 0.35 or fewer than 15 inliers are rejected to prevent bad stitches from corrupting the panorama.

---

### 6. Perspective Warping

Image 1 is warped onto a common canvas that accommodates both images. The canvas bounds are computed by projecting all four corners of image 1 through **H** and taking the bounding box over both images.

A translation offset is applied to handle negative coordinate shifts:

```python
translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
warped1 = cv2.warpPerspective(img1, translation @ H, (canvas_width, canvas_height))
```

---

### 7. Image Stitching & Blending

**Raw panorama (no blending):** A naive overlay where image 2 overwrites image 1 in the overlap region. Used only for diagnostic comparison.

**Distance-weighted blending:** Each pixel in the overlap region is blended proportionally to its distance from the nearest image boundary (computed via `cv2.distanceTransform`). This eliminates hard seams while preserving sharpness in non-overlapping regions.

```
blend_w1[overlap] = weight1[overlap] / (weight1[overlap] + weight2[overlap])
blend_w2[overlap] = 1 - blend_w1[overlap]
```

**Rectangular cropping:** After blending, the largest axis-aligned rectangle fully contained within valid (non-black) pixels is computed using a histogram-based maximal rectangle algorithm. A small inset (`inset=2` pixels) is applied to avoid border artifacts.

---

### 8. Multi-Image Stitching Across Clusters

For each cluster, the pipeline:

1. **Builds a pairwise match-score matrix** (count of good FLANN matches between every image pair).
2. **Determines a bidirectional ordering** starting from the two endpoints (lowest total match scores) and greedily expanding in both directions based on the highest match score at each step.
3. **Stitches sequentially** left-to-right using `stitch_pair`, with the accumulated panorama used as the left input at each step.
4. **Crops** the final panorama using `crop_panorama_rect`.

```python
panorama = cluster_imgs[0]
for i in range(1, len(cluster_imgs)):
    panorama = stitch_pair(panorama, cluster_imgs[i])
```

---

## Results

### Clustering

Images were successfully separated into 3 clusters corresponding to distinct scenes. Each cluster contained images with substantial color and scene similarity, confirming that histogram-based K-Means is sufficient for scene separation in this dataset.

---

### Feature Matching Comparison

| Matcher      | Speed    | Match Quality | Used in Pipeline |
|--------------|----------|---------------|------------------|
| Brute Force  | Slower   | Equivalent    | Diagnostic only  |
| FLANN        | Faster   | Equivalent    | Production       |

Both matchers produced comparable good-match counts after Lowe's ratio test. FLANN was adopted for the multi-stitching stage due to its superior speed on larger image sets.

---

### Panorama Quality

**Raw Panorama (No Blending):**  
A visible hard seam is present at the boundary between image 1 and image 2 due to exposure differences and the abrupt pixel transition.

**Distance-Weighted Blended Panorama:**  
The seam is eliminated. Transition zones blend smoothly, with full content preserved in non-overlapping regions.

**Final Cropped Panorama:**  
Black border regions from perspective warping are removed via the maximal valid-rectangle crop algorithm, producing a clean, rectangular output.

| Stage                         | Shape (example)          |
|-------------------------------|--------------------------|
| Warped canvas (pre-stitch)    | ~600 × 1400 px           |
| Blended panorama              | ~600 × 1400 px           |
| Cropped panorama              | ~560 × 1350 px           |

---

### Multi-Image Cluster Panoramas

Three panoramas were generated, one per cluster:

| Cluster | Images Stitched | Output File                              |
|---------|-----------------|------------------------------------------|
| 0       | N               | `panorama_outputs/panorama_cluster_0.png` |
| 1       | N               | `panorama_outputs/panorama_cluster_1.png` |
| 2       | N               | `panorama_outputs/panorama_cluster_2.png` |

All three stitches completed without homography failures, and the bidirectional ordering algorithm correctly resolved the left-to-right image sequence in each cluster.

---

## Output Files

| File / Directory                              | Description                                    |
|-----------------------------------------------|------------------------------------------------|
| `homography_matrix.csv`                       | 3×3 homography matrix (image1 → image2)       |
| `panorama_outputs/panorama_cluster_0.png`     | Final cropped panorama for Cluster 0          |
| `panorama_outputs/panorama_cluster_1.png`     | Final cropped panorama for Cluster 1          |
| `panorama_outputs/panorama_cluster_2.png`     | Final cropped panorama for Cluster 2          |

---

## Known Limitations

- **Pure translational assumption:** The bidirectional ordering heuristic assumes a roughly linear image sequence. It may fail for non-linear or circular panoramas.
- **Global color histograms for clustering:** Histogram features are sensitive to global illumination changes and may miscluster images with similar palettes from different scenes.
- **Sequential stitching accumulates error:** Stitching images one by one causes geometric drift. Bundle adjustment (e.g., via graph-based methods) would improve accuracy for large sets.
- **Single-scale blending:** Distance-weighted blending does not adapt to frequency content. Multi-band (Laplacian pyramid) blending would produce sharper results at fine scales.
- **RANSAC non-determinism:** Results may vary slightly across runs due to RANSAC's random sampling. Setting `cv2.setRNGSeed()` can make results reproducible.
