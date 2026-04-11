from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SINGLE_FEATURE_R2_THRESHOLD = 0.9995
SINGLE_FEATURE_CORR_THRESHOLD = 0.9995
PAIRWISE_R2_THRESHOLD = 0.999999
TOP_K = 10


def _safe_abs_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def _fit_and_score(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
    }


def audit_target_matrix(X, y, feature_names, target_name, run_pairwise=False, top_k=TOP_K):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    feature_names = list(feature_names)

    single_results = []
    for i, feature_name in enumerate(feature_names):
        xi = X[:, i].reshape(-1, 1)
        score = _fit_and_score(xi, y)
        score["feature"] = feature_name
        score["abs_corr"] = _safe_abs_corr(xi[:, 0], y)
        single_results.append(score)

    single_results.sort(key=lambda item: (item["r2"], item["abs_corr"]), reverse=True)
    critical_single = [
        item for item in single_results
        if item["r2"] >= SINGLE_FEATURE_R2_THRESHOLD or item["abs_corr"] >= SINGLE_FEATURE_CORR_THRESHOLD
    ]

    pair_results = []
    critical_pairs = []
    if run_pairwise:
        for i, j in combinations(range(len(feature_names)), 2):
            Xi = X[:, [i, j]]
            if np.nanstd(Xi[:, 0]) < 1e-12 and np.nanstd(Xi[:, 1]) < 1e-12:
                continue
            score = _fit_and_score(Xi, y)
            pair_result = {
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "r2": score["r2"],
                "mae": score["mae"],
                "rmse": score["rmse"],
                "coef_1": float(score["coef"][0]),
                "coef_2": float(score["coef"][1]),
                "intercept": score["intercept"],
            }
            pair_results.append(pair_result)

        pair_results.sort(key=lambda item: item["r2"], reverse=True)
        critical_pairs = [item for item in pair_results if item["r2"] >= PAIRWISE_R2_THRESHOLD]

    return {
        "target_name": target_name,
        "n_samples": int(len(y)),
        "n_features": int(len(feature_names)),
        "single_feature_r2_threshold": SINGLE_FEATURE_R2_THRESHOLD,
        "single_feature_corr_threshold": SINGLE_FEATURE_CORR_THRESHOLD,
        "pairwise_r2_threshold": PAIRWISE_R2_THRESHOLD,
        "top_single_features": single_results[:top_k],
        "critical_single_features": critical_single[:top_k],
        "pairwise_enabled": bool(run_pairwise),
        "top_feature_pairs": pair_results[:top_k],
        "critical_feature_pairs": critical_pairs[:top_k],
        "has_critical_findings": bool(critical_single or critical_pairs),
    }


def summarize_audit(label, report):
    lines = [f"[Leakage Audit] {label}: {report['n_features']} features, {report['n_samples']} samples."]

    if report["critical_single_features"]:
        feat = report["critical_single_features"][0]
        lines.append(
            "  Critical single-feature risk: "
            f"{feat['feature']} (R2={feat['r2']:.6f}, |corr|={feat['abs_corr']:.6f})"
        )

    if report.get("pairwise_enabled") and report["critical_feature_pairs"]:
        pair = report["critical_feature_pairs"][0]
        lines.append(
            "  Critical pairwise risk: "
            f"{pair['feature_1']} + {pair['feature_2']} "
            f"(R2={pair['r2']:.6f}, coeffs=[{pair['coef_1']:.3f}, {pair['coef_2']:.3f}])"
        )

    if not report["has_critical_findings"]:
        top = report["top_single_features"][0] if report["top_single_features"] else None
        if top is None:
            lines.append("  No features available for audit.")
        else:
            lines.append(
                "  No critical leakage found. Strongest single feature: "
                f"{top['feature']} (R2={top['r2']:.4f}, |corr|={top['abs_corr']:.4f})"
            )

    return "\n".join(lines)
