"""Paper-style visualization metrics and physical CFD metrics.

The TV term is reconstructed because the exact reference implementation is not
available. For a scalar field it is the mean absolute forward difference in x
plus the mean absolute forward difference in y. Vector channels are averaged.
"""
from __future__ import annotations
import numpy as np

TV_WEIGHT = 0.1


def paper_figure6_normalize(prediction, reference):
    """Normalize prediction and reference with prediction min/max."""
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    lower, upper = float(np.min(prediction)), float(np.max(prediction))
    scale = upper - lower
    if scale <= 1e-30:
        scale = 1.0
    return (prediction - lower) / scale, (reference - lower) / scale, lower, upper


def total_variation(field):
    field = np.asarray(field, dtype=float)
    if field.ndim == 2:
        field = field[..., None]
    return float(np.mean(np.abs(np.diff(field, axis=1))) +
                 np.mean(np.abs(np.diff(field, axis=0))))


def paper_style_metrics(prediction, reference):
    predicted, target, lower, upper = paper_figure6_normalize(prediction, reference)
    mse = float(np.mean((predicted - target) ** 2))
    tv_error = total_variation(predicted - target)
    return {"mse": mse, "tv": tv_error, "mse_tv": mse + TV_WEIGHT * tv_error,
            "prediction_min": lower, "prediction_max": upper,
            "normalization": "paper_figure6", "tv_definition": "reconstructed_forward_difference"}


def gauge_pressure(pressure, outlet_mask=None):
    pressure = np.asarray(pressure, dtype=float)
    gauge = float(np.mean(pressure[outlet_mask])) if outlet_mask is not None else float(np.mean(pressure))
    return pressure - gauge


def relative_error(prediction, reference):
    return float(np.linalg.norm(np.asarray(prediction)-np.asarray(reference)) /
                 (np.linalg.norm(reference)+1e-30))


def physical_metrics(predicted_u, reference_u, predicted_p, reference_p,
                     inlet_mask=None, outlet_mask=None):
    predicted_p = gauge_pressure(predicted_p, outlet_mask)
    reference_p = gauge_pressure(reference_p, outlet_mask)
    result = {"relative_velocity_error": relative_error(predicted_u[..., :2], reference_u[..., :2]),
              "relative_pressure_error": relative_error(predicted_p, reference_p)}
    if inlet_mask is not None and outlet_mask is not None:
        pred_drop = float(np.mean(predicted_p[inlet_mask])-np.mean(predicted_p[outlet_mask]))
        ref_drop = float(np.mean(reference_p[inlet_mask])-np.mean(reference_p[outlet_mask]))
        ratio = pred_drop/(ref_drop+1e-30)
        result.update({"predicted_pressure_drop": pred_drop, "reference_pressure_drop": ref_drop,
                       "pressure_drop_ratio": ratio, "pressure_drop_relative_error": abs(ratio-1.0)})
    return result
