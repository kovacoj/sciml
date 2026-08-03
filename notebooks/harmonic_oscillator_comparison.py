# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        from pyodide.http import open_url
        running_in_browser = True
    except ImportError:
        open_url = None
        running_in_browser = False

    return json, mo, np, open_url, plt, running_in_browser


@app.cell
def _(mo):
    mo.md(r"""
    # Harmonic oscillator solver comparison

    Explore precomputed eigenfunctions from direct finite elements, an explicit
    weak Firedrake-PyTorch solver, a strong PINN, and neural Rayleigh-Ritz.
    Training does not run in the browser; this notebook loads the same versioned
    result artifact used by the Fumadocs experiment page.
    """)
    return


@app.cell
def _(json, open_url, running_in_browser):
    if running_in_browser:
        with open_url("../../data/harmonic-oscillator-comparison.json") as stream:
            spectrum = json.load(stream)
    else:
        with open("public/data/harmonic-oscillator-comparison.json", encoding="utf-8") as stream:
            spectrum = json.load(stream)
    return (spectrum,)


@app.cell
def _(mo, spectrum):
    keys = list(spectrum["methods"])
    method = mo.ui.dropdown(
        {spectrum["methods"][key]["label"]: key for key in keys},
        value=spectrum["methods"][keys[0]]["label"],
        label="Solver",
    )
    state = mo.ui.slider(
        0,
        min(len(spectrum["methods"][key]["states"]) for key in keys) - 1,
        value=0,
        label="Eigenstate n",
        show_value=True,
    )
    mo.hstack([method, state], justify="center", gap=4)
    return method, state


@app.cell
def _(method, np, plt, spectrum, state):
    solver = spectrum["methods"][method.value]
    result = solver["states"][int(state.value)]
    x = np.asarray(spectrum["coordinates"])
    energy = float(result["eigenvalue"])
    psi = np.asarray(result["values"])

    figure, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    axis.plot(x, 0.5 * x**2, color="#2f2b27", label=r"$V(x)=x^2/2$")
    for exact in spectrum["exactEigenvalues"]:
        axis.hlines(exact, -0.42, 0.42, color="#8a8175", alpha=0.35)
    axis.plot(x, energy + 0.72 * psi, color=solver["color"], linewidth=2.2)
    axis.axhline(energy, color=solver["color"], linestyle="--", alpha=0.5)
    axis.set(xlim=spectrum["domain"], ylim=(0, spectrum["exactEigenvalues"][-1] + 1.5))
    axis.set_xlabel("position x")
    axis.set_ylabel("energy / shifted eigenfunction")
    axis.set_title(f'{solver["label"]}, n={state.value}, E={energy:.7f}')
    axis.grid(alpha=0.16)
    figure
    return


if __name__ == "__main__":
    app.run()
