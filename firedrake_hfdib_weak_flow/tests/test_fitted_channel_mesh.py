from src.fitted_channel_mesh import write_fitted_topology, write_fitted_topology_a


def test_fitted_topology_a_mesh_has_all_boundary_tags(tmp_path):
    text = write_fitted_topology_a(tmp_path / "a.msh").read_text()
    assert '1 1 "inlet"' in text
    assert '1 2 "outlet"' in text
    assert '1 3 "walls"' in text
    assert text.count(" 1 2 1 1 ") > 0
    assert text.count(" 1 2 2 2 ") > 0
    assert text.count(" 1 2 3 3 ") > 0


def test_gallery_meshes_have_all_boundary_tags(tmp_path):
    for name in "BCDEF":
        text = write_fitted_topology(tmp_path / f"{name}.msh", name).read_text()
        for marker in (1, 2, 3):
            assert f" 1 2 {marker} {marker} " in text
