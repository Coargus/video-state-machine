import unittest
from unittest.mock import Mock, patch

from vsm.checker.stormpy import StormModelChecker
from vsm.state.video_state import VideoState


class TestStormModelChecker(unittest.TestCase):
    def setUp(self):
        self.proposition_set = ["prop1", "prop2"]
        self.ltl_formula = "F(prop1 & prop2)"  # Update this line
        self.checker = StormModelChecker(self.proposition_set, self.ltl_formula)

    def test_init(self):
        self.assertEqual(self.checker.proposition_set, self.proposition_set)
        self.assertEqual(self.checker.ltl_formula, self.ltl_formula)
        self.assertFalse(self.checker.is_filter)
        self.assertFalse(self.checker.verbose)

    @patch("stormpy.SparseModelComponents")
    @patch("stormpy.SparseMA")
    def test_create_model_ma(self, mock_sparse_ma, mock_components):
        transitions = [(0, 1, 0.5), (1, 2, 1.0)]
        states = [
            Mock(spec=VideoState, descriptive_label=["prop1"], state_index=i)
            for i in range(3)
        ]

        model = self.checker.create_model(transitions, states, model_type="ma")

        self.assertTrue(mock_components.called)
        self.assertTrue(mock_sparse_ma.called)
        self.assertEqual(model, mock_sparse_ma.return_value)

    @patch("stormpy.storage.SparseDtmc")
    def test_create_model_dtmc(self, mock_sparse_dtmc):
        transitions = [(0, 1, 0.5), (1, 2, 1.0)]
        states = [
            Mock(spec=VideoState, descriptive_label=["prop2"], state_index=i)
            for i in range(3)
        ]

        model = self.checker.create_model(
            transitions, states, model_type="dtmc"
        )

        self.assertTrue(mock_sparse_dtmc.called)
        self.assertEqual(model, mock_sparse_dtmc.return_value)

    @patch("stormpy.model_checking")
    @patch.object(StormModelChecker, "create_model")
    def test_check_automaton(self, mock_create_model, mock_model_checking):
        transitions = [(0, 1, 0.5), (1, 2, 1.0)]
        states = [Mock(spec=VideoState) for _ in range(3)]
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        result = self.checker.check_automaton(
            transitions, states, model_type="dtmc"
        )

        self.assertTrue(mock_create_model.called)
        self.assertTrue(mock_model_checking.called)
        self.assertEqual(result, mock_model_checking.return_value)

    def test_verification_result_eval(self):
        mock_result = Mock()
        mock_result.__str__ = Mock(return_value="{true}")
        self.assertTrue(self.checker._verification_result_eval(mock_result))

        mock_result.__str__ = Mock(return_value="{false}")
        self.assertFalse(self.checker._verification_result_eval(mock_result))

        mock_result.__str__ = Mock(return_value="{true, false}")
        self.assertTrue(self.checker._verification_result_eval(mock_result))


if __name__ == "__main__":
    unittest.main()
