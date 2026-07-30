from types import SimpleNamespace

import numpy as np

from af3_hallucination.af3.consistency import scatter_pseudo_beta


def test_pseudo_beta_scatter_uses_only_official_gather_layout():
    info = SimpleNamespace(
        pseudo_beta_info=SimpleNamespace(
            token_atoms_to_pseudo_beta=SimpleNamespace(
                gather_idxs=np.asarray([1, 4]),
                gather_mask=np.asarray([True, True]),
                input_shape=np.asarray([2, 3]),
            )
        )
    )
    pseudo_beta = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    dense = scatter_pseudo_beta(info, pseudo_beta)
    assert dense.shape == (2, 3, 3)
    assert np.array_equal(dense.reshape((-1, 3))[[1, 4]], pseudo_beta)
    assert np.count_nonzero(dense) == 6
