# Reconstruction notes

- Chamber (area of interest), cell size, port positions and port widths are
  recovered from the published dataset (`tpfm_unet/data/coordinates_64.csv`,
  statistics over `data/mixer_64.npz`); see ../../references/UPSTREAM.md.
- The port extension length outside the area of interest is NOT published;
  it is reconstructed here as 0.016 m (8 cells). The authors' complete
  OpenFOAM case (blockMeshDict, solver settings) was not available.
- This case is therefore a reconstruction, not the authors' original case.
