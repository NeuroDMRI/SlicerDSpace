"""Compare independently generated Python/MRtrix connectomes and node ordering."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node_matrix(outputs, metric):
    with Path(outputs["nodes_path"]).open(newline="") as stream:
        labels = [int(row["label"]) for row in csv.DictReader(stream, delimiter="\t")]
    with Path(outputs["matrix_path"]).open(newline="") as stream:
        header = next(csv.reader(stream))
    if header[:2] == ["source_node", "target_node"]:
        index = {label: position for position, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)))
        with Path(outputs["matrix_path"]).open(newline="") as stream:
            for row in csv.DictReader(stream):
                i, j = index[int(row["source_node"])], index[int(row["target_node"])]
                matrix[i, j] = matrix[j, i] = float(row[metric])
        return matrix
    dense = np.loadtxt(outputs["matrix_path"], delimiter=",")
    assert dense.ndim == 2 and dense.shape[0] == dense.shape[1]
    assert np.isfinite(dense).all() and (dense >= 0).all()
    indices = list(range(len(labels))) if dense.shape[0] == len(labels) else [label - 1 for label in labels]
    assert all(0 <= index < dense.shape[0] for index in indices)
    return dense[np.ix_(indices, indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--mrtrix-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = {}
    for backend, path in (("python", args.python_manifest), ("mrtrix", args.mrtrix_manifest)):
        references[backend] = {task["name"]: task for task in json.loads(path.read_text())["tasks"]}
    results = []
    for name in ("connectome.streamline_count", "connectome.sift2_weighted"):
        files = {backend: tasks[name]["outputs"] for backend, tasks in references.items()}
        python_qc = json.loads(Path(references["python"][name]["qc_path"]).read_text())
        mrtrix_qc = json.loads(Path(references["mrtrix"][name]["qc_path"]).read_text())
        matrices = {backend: node_matrix(outputs, python_qc["metric"]) for backend, outputs in files.items()}
        assert str(python_qc["backend"]).startswith("python"), python_qc["backend"]
        assert mrtrix_qc["backend"] == "mrtrix3.tck2connectome", mrtrix_qc["backend"]
        assert all(matrix.shape == (84, 84) and np.isfinite(matrix).all()
                   and np.allclose(matrix, matrix.T, rtol=0, atol=1e-8)
                   and np.all(np.diag(matrix) == 0) for matrix in matrices.values())
        assert Path(files["python"]["nodes_path"]).read_bytes() == Path(files["mrtrix"]["nodes_path"]).read_bytes()
        difference = np.abs(matrices["python"] - matrices["mrtrix"])
        # MRtrix 3.0.8 uses float accumulation for high-valued label indices.
        # Assignment must be exact; weighted sums allow float32 accumulation.
        rtol, atol = (0., 0.) if name.endswith("streamline_count") else (1e-5, 1e-7)
        same_edges = bool(np.array_equal(matrices["python"] > 0, matrices["mrtrix"] > 0))
        nonzero = matrices["python"] > 0
        relative = difference[nonzero] / matrices["python"][nonzero]
        passed = same_edges and bool(np.allclose(matrices["python"], matrices["mrtrix"], rtol=rtol, atol=atol))
        results.append({"task": name, "passed": passed, "shape": [84, 84],
                        "max_absolute_difference": float(difference.max()), "max_relative_difference": float(relative.max()),
                        "rtol": rtol, "atol": atol, "nonzero_edge_pattern_identical": same_edges,
                        "node_order_identical": True,
                        "matrix_sources": {backend: {"path": outputs["matrix_path"],
                                            "sha256": digest(outputs["matrix_path"])}
                                           for backend, outputs in files.items()}})
    result = {"passed": all(row["passed"] for row in results), "comparisons": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
