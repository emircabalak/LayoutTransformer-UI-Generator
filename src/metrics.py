import numpy as np
from config import NUM_CLASSES


def compute_iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = a1 + a2 - inter

    if union > 0:
        return inter / union
    return 0.0


def compute_overlap_rate(elements):
    bboxes = []
    for e in elements:
        bboxes.append([e[1], e[2], e[3], e[4]])

    if len(bboxes) < 2:
        return 0.0

    total = 0
    overlapping = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            total += 1
            iou = compute_iou(bboxes[i], bboxes[j])
            if iou > 0.01:
                overlapping += 1

    return overlapping / total


def compute_alignment_score(elements, threshold=0.02):
    bboxes = []
    for e in elements:
        bboxes.append([e[1], e[2], e[3], e[4]])

    if len(bboxes) < 2:
        return 1.0

    edges = {"left": [], "right": [], "top": [], "bottom": []}
    for b in bboxes:
        edges["left"].append(b[0])
        edges["top"].append(b[1])
        edges["right"].append(b[2])
        edges["bottom"].append(b[3])

    aligned = 0
    total = 0
    for vals in edges.values():
        for i in range(len(vals)):
            total += 1
            for j in range(len(vals)):
                if i == j:
                    continue
                if abs(vals[i] - vals[j]) < threshold:
                    aligned += 1
                    break

    if total > 0:
        return aligned / total
    return 1.0


def compute_validity(elements):
    if not elements:
        return 0.0

    valid = 0
    for e in elements:
        cls_id = int(e[0])
        x1, y1, x2, y2 = e[1], e[2], e[3], e[4]
        is_valid_x = (0 <= x1 < x2 <= 1)
        is_valid_y = (0 <= y1 < y2 <= 1)
        is_valid_cls = (0 <= cls_id < NUM_CLASSES)
        if is_valid_x and is_valid_y and is_valid_cls:
            valid += 1

    return valid / len(elements)


def compute_class_diversity(elements):
    if not elements:
        return 0
    classes = set()
    for e in elements:
        classes.add(int(e[0]))
    return len(classes)


def evaluate_layouts(layouts):
    overlap_rates = []
    alignment_scores = []
    validity_scores = []
    elem_counts = []
    class_diversities = []

    for layout in layouts:
        if not layout:
            continue
        overlap_rates.append(compute_overlap_rate(layout))
        alignment_scores.append(compute_alignment_score(layout))
        validity_scores.append(compute_validity(layout))
        elem_counts.append(len(layout))
        class_diversities.append(compute_class_diversity(layout))

    n = len(overlap_rates)
    if n == 0:
        return {"num_samples": 0}

    return {
        "num_samples": n,
        "num_empty": len(layouts) - n,
        "avg_elements": np.mean(elem_counts),
        "avg_overlap_rate": np.mean(overlap_rates),
        "std_overlap_rate": np.std(overlap_rates),
        "avg_alignment_score": np.mean(alignment_scores),
        "std_alignment_score": np.std(alignment_scores),
        "avg_validity": np.mean(validity_scores),
        "avg_class_diversity": np.mean(class_diversities),
    }


def print_metrics(results, model_name="LayoutTransformer"):
    print(f"\n{'='*55}")
    print(f"  {model_name} — Degerlendirme Sonuclari")
    print(f"{'='*55}")
    print(f"  Ornek sayisi:            {results['num_samples']}")
    if results.get('num_empty', 0) > 0:
        print(f"  Bos layout:              {results['num_empty']}")
    print(f"  Ort. eleman sayisi:      {results.get('avg_elements', 0):.1f}")
    print(f"  Ort. Overlap Rate:       {results['avg_overlap_rate']:.4f} "
          f"(+/-{results['std_overlap_rate']:.4f})")
    print(f"  Ort. Alignment Score:    {results['avg_alignment_score']:.4f} "
          f"(+/-{results['std_alignment_score']:.4f})")
    print(f"  Ort. Validity:           {results.get('avg_validity', 0):.4f}")
    print(f"  Ort. Class Diversity:    {results.get('avg_class_diversity', 0):.1f}")
    print(f"{'='*55}")
