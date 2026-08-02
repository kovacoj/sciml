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
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Dirichlet Laplacian Eigenmode Explorer

    This interactive notebook studies

    \[
    -u''(x) = \lambda u(x),
    \qquad x\in(0,1),
    \qquad u(0)=u(1)=0.
    \]

    Its exact eigenpairs are

    \[
    \lambda_n=(n\pi)^2,
    \qquad
    u_n(x)=\sqrt{2}\sin(n\pi x).
    \]

    The numerical approximation below uses a centered finite-difference
    discretization on a uniform grid. Change the mode number and grid
    resolution to inspect the approximation and its error.

    > This is a deterministic mathematical demonstration. It does not
    > present results from the research experiments documented elsewhere
    > on the website.
    """)
    return


@app.cell
def _(mo):
    mode_selector = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=1,
        label="Eigenmode n",
        show_value=True,
    )

    grid_selector = mo.ui.slider(
        start=20,
        stop=200,
        step=10,
        value=80,
        label="Interior grid points N",
        show_value=True,
    )

    mo.hstack([mode_selector, grid_selector])
    return grid_selector, mode_selector


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _(grid_selector, mode_selector, np):
    grid_size = int(grid_selector.value)
    mode_number = int(mode_selector.value)

    mesh_width = 1.0 / (grid_size + 1)
    interior_points = np.arange(1, grid_size + 1, dtype=float) * mesh_width

    main_diagonal = 2.0 * np.ones(grid_size) / mesh_width**2
    off_diagonal = -1.0 * np.ones(grid_size - 1) / mesh_width**2

    discrete_laplacian = (
        np.diag(main_diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    )

    discrete_eigenvalues, discrete_eigenvectors = np.linalg.eigh(
        discrete_laplacian
    )

    numerical_eigenvalue = float(discrete_eigenvalues[mode_number - 1])
    numerical_mode = discrete_eigenvectors[:, mode_number - 1].copy()

    exact_eigenvalue = float((mode_number * np.pi) ** 2)
    exact_mode_interior = np.sqrt(2.0) * np.sin(
        mode_number * np.pi * interior_points
    )

    numerical_norm = np.sqrt(mesh_width * np.sum(numerical_mode**2))
    numerical_mode /= numerical_norm

    weighted_overlap = mesh_width * np.dot(
        numerical_mode, exact_mode_interior
    )
    if weighted_overlap < 0.0:
        numerical_mode *= -1.0

    relative_eigenvalue_error = abs(
        numerical_eigenvalue - exact_eigenvalue
    ) / exact_eigenvalue

    eigenfunction_l2_error = np.sqrt(
        mesh_width
        * np.sum((numerical_mode - exact_mode_interior) ** 2)
    )

    full_grid = np.concatenate(([0.0], interior_points, [1.0]))
    exact_mode_full = np.sqrt(2.0) * np.sin(
        mode_number * np.pi * full_grid
    )
    numerical_mode_full = np.concatenate(([0.0], numerical_mode, [0.0]))

    convergence_sizes = np.array([10, 20, 40, 80, 160, 320], dtype=int)
    convergence_widths = 1.0 / (convergence_sizes + 1)

    # Closed-form eigenvalue of the centered finite-difference operator.
    convergence_eigenvalues = (
        4.0
        / convergence_widths**2
        * np.sin(
            0.5 * mode_number * np.pi * convergence_widths
        )
        ** 2
    )

    convergence_errors = np.abs(
        convergence_eigenvalues - exact_eigenvalue
    ) / exact_eigenvalue
    return (
        convergence_errors,
        convergence_sizes,
        eigenfunction_l2_error,
        exact_eigenvalue,
        exact_mode_full,
        full_grid,
        grid_size,
        mode_number,
        numerical_eigenvalue,
        numerical_mode_full,
        relative_eigenvalue_error,
    )


@app.cell
def _(
    eigenfunction_l2_error,
    exact_eigenvalue,
    grid_size,
    mo,
    mode_number,
    numerical_eigenvalue,
    relative_eigenvalue_error,
):
    mo.md(
        f"""
        ## Current approximation

        | Quantity | Value |
        |---|---:|
        | Mode number | `{mode_number}` |
        | Interior grid points | `{grid_size}` |
        | Exact eigenvalue | `{exact_eigenvalue:.10f}` |
        | Discrete eigenvalue | `{numerical_eigenvalue:.10f}` |
        | Relative eigenvalue error | `{relative_eigenvalue_error:.3e}` |
        | Weighted L2 eigenfunction error | `{eigenfunction_l2_error:.3e}` |
        """
    )
    return


@app.cell
def _(
    convergence_errors,
    convergence_sizes,
    exact_mode_full,
    full_grid,
    mode_number,
    numerical_mode_full,
    plt,
):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.0, 4.2))

    mode_axis = axes[0]
    mode_axis.plot(
        full_grid,
        exact_mode_full,
        label="Exact eigenfunction",
        linewidth=2.0,
    )
    mode_axis.plot(
        full_grid,
        numerical_mode_full,
        "o--",
        label="Finite-difference eigenvector",
        markersize=3.0,
    )
    mode_axis.set_xlabel("x")
    mode_axis.set_ylabel("u(x)")
    mode_axis.set_title(f"Eigenmode n = {mode_number}")
    mode_axis.grid(True, alpha=0.25)
    mode_axis.legend()

    convergence_axis = axes[1]
    convergence_axis.loglog(
        convergence_sizes,
        convergence_errors,
        "o-",
        label="Relative eigenvalue error",
    )

    reference_scale = convergence_errors[0] * (
        convergence_sizes[0] / convergence_sizes
    ) ** 2
    convergence_axis.loglog(
        convergence_sizes,
        reference_scale,
        "--",
        label="Second-order reference",
    )

    convergence_axis.set_xlabel("Interior grid points N")
    convergence_axis.set_ylabel("Relative error")
    convergence_axis.set_title("Eigenvalue convergence")
    convergence_axis.grid(True, which="both", alpha=0.25)
    convergence_axis.legend()

    figure.tight_layout()
    figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    The centered finite-difference discretization is second-order accurate
    for this smooth eigenproblem. Higher modes require finer grids because
    their eigenfunctions oscillate more rapidly.

    The browser application is intentionally small. Full finite-element,
    PINN, optimization, and CFD experiments should be executed outside the
    browser and published through compact, curated result files.
    """)
    return


if __name__ == "__main__":
    app.run()
