"""Unit tests for persistence module."""

import numpy as np
from romanisim import persistence
from romanisim.models import parameters


def test_fermi():
    # fermi(x, dt, A, x0, dx, alpha, gamma)
    # x, dt, A, x0, dx, alpha, gamma, answer
    answers = [(1, 1000, 1, 1, 1, 1, 1, 0.5),
               (100, 1000, 1, 1, 1, 1, 1, 100),
               (100, 1000, 2, 1, 1, 1, 1, 200),
               (100, 1000, 1, 1, 1, 2, 1, 10000),
               (100, 1000, 1, 100, 1, 1, 1, 0.5),
               (100, 1000, 1, 1, 1, 1, 2, 100),
               (100, 2000, 1, 1, 1, 1, 1, 50),
               (100, 2000, 1, 1, 1, 1, 2, 25),
               (0, 1000, 1, 1, 1, 1, 1, 0),
               ]
    hw = parameters.persistence['half_well']
    parameters.persistence['half_well'] = 0.1
    for answer in answers:
        rate = persistence.fermi(*answer[:-1])
        assert np.isclose(rate, answer[-1])
        persist = persistence.Persistence(
            answer[0], 0, 0, *answer[2:-1])
        rate2 = persist.current((answer[1]) / 60 / 60 / 24)
        assert np.isclose(rate, rate2)
    parameters.persistence['half_well'] = hw
    hwpersistence = persistence.fermi(hw, 1000, 1, hw, 100, 1, 1)
    linpersistence = persistence.fermi(hw / 2, 1000, 1, hw, 100, 1, 1)
    assert np.isclose(hwpersistence / 2, linpersistence)


def test_persistence(tmp_path):
    persist = persistence.Persistence()
    img = np.zeros((100, 100), dtype='i4')
    img2 = img.copy()
    persist.add_to_read(img2, 0)
    assert np.all(img == img2)
    # there were no persistence-affected pixels in this image
    img2[:, :] = 10**5
    persist.update(img2, 0)
    img3 = img.copy()
    persist.add_to_read(img3, 1 / 60 / 60 / 24)
    assert np.all(img3 > img)
    img4 = img.copy()
    persist.add_to_read(img4, 1)  # one day later
    assert np.all(img4 < img3)  # persistence decreases with time
    assert len(persist.x) == img.reshape(-1).shape[0]
    # all pixels were affected
    persist.update(img2, 1 / 60 / 60 / 24)
    assert len(persist.x) == img.reshape(-1).shape[0] * 2
    # all pixels got zapped twice
    persist.update(img3, 1)
    assert len(persist.x) == 0  # one day later, everything is okay again.
    fn = tmp_path / 'persist.asdf'
    persist.write(fn)
    persist2 = persistence.Persistence.read(fn)
    dd = persist.to_dict()
    for name in dd:
        assert np.all(getattr(persist, name) == getattr(persist2, name))
