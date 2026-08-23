from pathlib import Path

from src.brinkman_mesh import write_segmented_square_mesh


def test_segmented_gmsh_contains_port_and_wall_markers(tmp_path):
    path = write_segmented_square_mesh(tmp_path / "ports.msh", 32)
    text = path.read_text()
    assert '1 1 "inlet"' in text
    assert '1 2 "outlet"' in text
    assert '1 3 "walls"' in text
    assert text.count(" 1 2 1 1 ") > 0
    assert text.count(" 1 2 2 2 ") > 0
    assert text.count(" 1 2 3 3 ") > 0
