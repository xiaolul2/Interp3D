import numpy as np
from sklearn.neighbors import NearestNeighbors

def farthest_point_sampling(points, num_samples):
    N, _ = points.shape
    sampled_idx = [np.random.randint(0, N)]
    distances = np.full(N, np.inf)
    for _ in range(1, num_samples):
        current_point = points[sampled_idx[-1]]
        dist = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist)
        sampled_idx.append(np.argmax(distances))
    return np.array(sampled_idx)

def svd_rigid_align(A, B):
    A_mean = A.mean(0)
    B_mean = B.mean(0)
    A_centered = A - A_mean
    B_centered = B - B_mean
    H = B_centered.T @ A_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    return R, A_mean, B_mean

def interpolate_with_fps_rigid_control(centers_A, centers_B, center_interp, alpha, num_anchors=100, k=8):
    fps_idx = farthest_point_sampling(center_interp, num_anchors)
    anchors = center_interp[fps_idx]
    anchor_assigner = NearestNeighbors(n_neighbors=1).fit(anchors)
    patch_ids = anchor_assigner.kneighbors(center_interp, return_distance=False).squeeze()
    corrected = np.zeros_like(center_interp)
    nbrs_A = NearestNeighbors(n_neighbors=k).fit(centers_A)
    nbrs_B = NearestNeighbors(n_neighbors=k).fit(centers_B)

    for i in range(num_anchors):
        mask = (patch_ids == i)
        if np.sum(mask) < k:
            corrected[mask] = center_interp[mask]
            continue
        patch_center = anchors[i].reshape(1, 3)
        idx_A = nbrs_A.kneighbors(patch_center, return_distance=False)[0]
        idx_B = nbrs_B.kneighbors(patch_center, return_distance=False)[0]
        patch_A = centers_A[idx_A]
        patch_B = centers_B[idx_B]
        R, A_mean, B_mean = svd_rigid_align(patch_A, patch_B)
        corrected[mask] = ((center_interp[mask] - A_mean) @ R.T) + B_mean
    return corrected