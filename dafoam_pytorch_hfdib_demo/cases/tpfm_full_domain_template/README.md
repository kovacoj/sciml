# TPFM full-domain reconstruction template

This case is generated from `cases/four_port_64x64` by
`tpfm_reference.full_domain_case`. It adds equal fixed inlet/outlet channel
extensions while retaining the central 64x64 ROI at exactly `h=0.002 m`.

Predeclared extension candidates are 8, 16, and 32 cells per side. Sample 0
selects one candidate; samples 274 and 549 remain untouched validation cases.
The topology varies only in the ROI. Outside it, only the two fixed 8-cell-wide
port channels are fluid.
