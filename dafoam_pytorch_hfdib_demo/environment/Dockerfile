FROM dafoam/opt-packages:v5.0.0

USER dafoamuser

SHELL ["/bin/bash", "-lc"]

# loadDAFoam.sh puts the image's Miniconda python on PATH (there is no bare
# python3 before sourcing it).  The venv layers PyTorch on top with
# --system-site-packages so DAFoam/PETSc/MPI/OpenMDAO are shared, NOT
# reinstalled.
RUN source $HOME/dafoam/loadDAFoam.sh && \
    python3 -m venv --system-site-packages $HOME/venvs/dafoam-torch

# Install PyTorch into the isolated venv.
# Do not upgrade numpy, mpi4py, petsc4py, OpenMDAO, or DAFoam.
RUN source $HOME/venvs/dafoam-torch/bin/activate && \
    python -m pip install --upgrade pip && \
    python -m pip install torch

# One-stop activation: DAFoam first, then the venv (venv must win PATH).
RUN printf '%s\n' \
        'source $HOME/dafoam/loadDAFoam.sh' \
        'source $HOME/venvs/dafoam-torch/bin/activate' \
        > $HOME/activate_dafoam_torch.sh

# The container runs bind-mounted workloads as the HOST uid (not 1002):
# make the home tree traversable so bash activation scripts are readable.
USER root
RUN chmod 0755 /home/dafoamuser && \
    chmod a+r /home/dafoamuser/activate_dafoam_torch.sh && \
    chmod -R a+rX /home/dafoamuser/venvs
USER dafoamuser

WORKDIR /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
