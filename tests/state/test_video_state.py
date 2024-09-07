import pytest

from vsm.state.video_state import VideoState


@pytest.fixture()
def init_state():
    return VideoState(
        state_index=0,
        frame_index=-1,
        label="init",
        proposition_set=["object 1", "object 0"],
    )


def test_get_descriptive_label(init_state):
    assert init_state._get_descriptive_label("init") == ["init"]
    assert init_state._get_descriptive_label("TT") == ["object 1", "object 0"]
    assert init_state._get_descriptive_label("TF") == ["object 1"]
    assert init_state._get_descriptive_label("FT") == ["object 0"]
    assert init_state._get_descriptive_label("FF") == []


def test_update(init_state):
    init_state.update(0, "TT")
    assert init_state.frame_index == 0
    assert init_state.label == "TT"
    assert init_state.descriptive_label == ["object 1", "object 0"]
    init_state.update(1, "TF")
    assert init_state.frame_index == 1
    assert init_state.label == "TF"
    assert init_state.descriptive_label == ["object 1"]
    init_state.update(2, "FT")
    assert init_state.frame_index == 2
    assert init_state.label == "FT"
    assert init_state.descriptive_label == ["object 0"]


def test_compute_probability_1(init_state):
    # test case: TT
    object_1_probabilities = [0.8]
    object_2_probabilities = [0.5]
    init_state.update(0, "TT")
    init_state.compute_probability(
        probabilities=[object_1_probabilities, object_2_probabilities]
    )
    assert init_state.probability == 0.40  # 0.8 * 0.5


def test_compute_probability_2(init_state):
    # test case: TF
    object_1_probabilities = [0.8]
    object_2_probabilities = [0.5]
    init_state.update(0, "FT")
    init_state.compute_probability(
        probabilities=[object_1_probabilities, object_2_probabilities]
    )
    assert init_state.probability == 0.10  # (1-0.8) * 0.5
