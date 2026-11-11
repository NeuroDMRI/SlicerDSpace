import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional sklearn dependency helper
    threadpool_limits = None


def _parallel_exact_kmeans_labels(p, k: int, n_init: int = 10, random_state: int = 0):
    """Run sklearn's exact 1-thread KMeans initializations concurrently across workers."""
    import sklearn.cluster._kmeans as km_mod

    est = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    X = est._validate_data(
        p,
        accept_sparse="csr",
        dtype=[np.float64, np.float32],
        order="C",
        copy=est.copy_x,
        accept_large_sparse=False,
    )
    est._check_params_vs_input(X)
    rng = check_random_state(est.random_state)
    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    x_squared_norms = row_norms(X_centered, squared=True)

    kmeans_single = km_mod._kmeans_single_elkan if est._algorithm == "elkan" else km_mod._kmeans_single_lloyd
    seeds = rng.randint(np.iinfo(np.int32).max, size=est._n_init)

    env_cpus = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    avail = int(env_cpus) if env_cpus.isdigit() and int(env_cpus) > 0 else min(8, os.cpu_count() or 1)
    max_workers = max(1, min(int(n_init), avail))

    def _run_seed(seed_val):
        if threadpool_limits is None:
            centers_init = est._init_centroids(
                X_centered, x_squared_norms=x_squared_norms, init="k-means++",
                random_state=check_random_state(int(seed_val)), sample_weight=sample_weight,
            )
            return kmeans_single(
                X_centered, sample_weight, centers_init, max_iter=est.max_iter,
                verbose=est.verbose, tol=est._tol, n_threads=1,
            )
        with threadpool_limits(limits=1):
            centers_init = est._init_centroids(
                X_centered, x_squared_norms=x_squared_norms, init="k-means++",
                random_state=check_random_state(int(seed_val)), sample_weight=sample_weight,
            )
            return kmeans_single(
                X_centered, sample_weight, centers_init, max_iter=est.max_iter,
                verbose=est.verbose, tol=est._tol, n_threads=1,
            )

    if max_workers <= 1:
        results = [_run_seed(s) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_seed, seeds))

    best_labels = None
    best_inertia = None
    best_centers = None
    for labels, inertia, centers, _ in results:
        if best_inertia is None or (
            inertia < best_inertia and not km_mod._is_same_clustering(labels, best_labels, k)
        ):
            best_labels = labels
            best_centers = centers
            best_inertia = inertia

    distinct = len(set(best_labels))
    if distinct < k:
        raise ValueError("Fewer distinct clusters than k")
    return best_labels


def coordinate_clustering(p, nsl):
    if nsl > 5e5:
        k = int(round(p.shape[0] ** 0.5))
    else:
        k = 60
    t0 = time.time()
    try:
        labels = _parallel_exact_kmeans_labels(p, k, n_init=10, random_state=0)
    except Exception:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        if threadpool_limits is None:
            labels = km.fit_predict(p)
        else:
            with threadpool_limits(limits=1):
                labels = km.fit_predict(p)
    labels = labels + 1
    elapsed = [f"Time spent on surface coordinate clustering (k={k}):", _format_elapsed(time.time() - t0)]
    return labels, elapsed


def _format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
