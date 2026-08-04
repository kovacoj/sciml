FROM dafoam/opt-packages:v5.0.0

USER dafoamuser

RUN python3 -m venv \
        --system-site-packages \
        /home/dafoamuser/venvs/dafoam-torch

ENV PATH="/home/dafoamuser/venvs/dafoam-torch/bin:${PATH}"

# Install PyTorch into the isolated venv.
# Do not upgrade numpy, mpi4py, petsc4py, OpenMDAO, or DAFoam.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch

WORKDIR /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
