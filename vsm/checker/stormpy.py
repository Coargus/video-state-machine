import logging

import numpy as np
import stormpy
import stormpy.examples.files
from stormpy.core import ExplicitQualitativeCheckResult

from vsm.state.video_state import VideoState


class StormModelChecker:
    """Model checker using Storm for verifying properties."""

    def __init__(
        self,
        proposition_set: list[str],
        ltl_formula: str,
        verbose: bool = False,
        is_filter: bool = False,
    ) -> None:
        """Initialize the StormModelChecker.

        Args:
            proposition_set: List of propositions.
            ltl_formula: LTL formula to check.
            verbose: Enable verbose output.
            is_filter: Apply filtering to results.
        """
        self.proposition_set = proposition_set
        self.ltl_formula = ltl_formula
        self.is_filter = is_filter
        self.verbose = verbose

    def create_model(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[VideoState],
        model_type: str = "ma",
        verbose: bool = False,
    ) -> any:
        """Create model.

        Args:
            transitions (list[tuple[int, int, float]]): List of transitions.
            states (list[VideoState]): List of states.
            model_type (str): Type of model to create ("sparse_ma" or "dtmc").
            verbose (bool): Whether to print verbose output.
        """
        state_labeling = self._build_label_func(states, self.proposition_set)
        if model_type in ["sparse_ma", "mdp"]:
            transition_matrix = self._build_trans_matrix(
                transitions=transitions,
                states=states,
                model_type="nondeterministic",
            )
        else:
            transition_matrix = self._build_trans_matrix(
                transitions=transitions,
                states=states,
                model_type="deterministic",
            )
        components = stormpy.SparseModelComponents(
            transition_matrix=transition_matrix,
            state_labeling=state_labeling,
        )
        if model_type == "ma":
            markovian_states = stormpy.BitVector(
                len(states), list(range(len(states)))
            )
            components.markovian_states = markovian_states
            components.exit_rates = [0.0 for _ in range(len(states))]
            model = stormpy.SparseMA(components)
        elif model_type == "dtmc":
            model = stormpy.storage.SparseDtmc(components)
        elif model_type == "mdp":
            model = stormpy.storage.SparseMdp(components)
        else:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        if verbose:
            print(transition_matrix)  # noqa: T201
            print(model)  # noqa: T201
        return model

    def check_automaton(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[VideoState],
        verbose: bool = False,
        model_type: str = "ma",
        is_filter: bool = False,
    ) -> any:
        """Check automaton.

        Args:
            transitions: List of transitions.
            states: List of states.
            verbose: Enable verbose output.
            is_filter: Apply filtering to results.
        """
        model = self.create_model(
            transitions=transitions,
            states=states,
            model_type=model_type,
            verbose=verbose,
        )
        # Check the model (Markov Automata)
        if model_type == "ma":
            result = self._model_checking(model, self.ltl_formula, is_filter)
            # return self._verification_result_eval(verification_result=result)  # noqa: ERA001
        elif model_type == "dtmc":
            properties = stormpy.parse_properties(self.ltl_formula)
            result = stormpy.model_checking(model, properties[0])
        return result

    def _verification_result_eval(
        self, verification_result: ExplicitQualitativeCheckResult
    ) -> bool:
        # string result is "true" when is absolutely true
        # but it returns "true, false" when we have some true and false
        verification_result_str = str(verification_result)
        string_result = verification_result_str.split("{")[-1].split("}")[0]
        if len(string_result) == 4:
            if string_result[0] == "t":  # 0,6
                result = True
        elif len(string_result) > 5:
            # "true, false" -> some true and some false
            result = True
        else:
            result = False
        return result

    def _model_checking(
        self,
        model: stormpy.storage.SparseMA,
        formula_str: str,
        is_filter: bool = False,
    ) -> any:
        """Model checking.

        Args:
            model: Markov Automata.
            formula_str: Formula string.
            is_filter: Apply filtering to results.

        Returns:
            any: Result.
        """
        # Initialize Prism Program
        path = stormpy.examples.files.prism_dtmc_die  #  prism_mdp_maze
        prism_program = stormpy.parse_prism_program(path)

        # Define Properties
        properties = stormpy.parse_properties(formula_str, prism_program)

        # Get Result and Filter it
        result = stormpy.model_checking(model, properties[0])

        if is_filter:
            filter = stormpy.create_filter_initial_states_sparse(model)
            result.filter(filter)

        return result

    def _build_trans_matrix(
        self,
        transitions: list[tuple[int, int, float]],
        states: list[VideoState],
        model_type: str = "nondeterministic",
    ) -> stormpy.storage.SparseMatrix:
        """Build transition matrix.

        Args:
            transitions: List of transitions.
            states: List of states.
            model_type: Type of model ("nondeterministic" or "deterministic").
        """
        if model_type == "nondeterministic":
            matrix = np.zeros((len(states), len(states)))
            for t in transitions:
                matrix[int(t[0]), int(t[1])] = float(t[2])
            trans_matrix = stormpy.build_sparse_matrix(
                matrix, list(range(len(states)))
            )

        elif model_type == "deterministic":
            num_states = len(states)
            builder = stormpy.SparseMatrixBuilder(
                rows=len(states),
                columns=len(states),
                entries=len(transitions),
                force_dimensions=False,
            )
            # Create a set of all states that have outgoing transitions
            states_with_transitions = set(src for src, _, _ in transitions)

            for src, dest, prob in transitions:
                builder.add_next_value(src, dest, prob)

            # Add self-loops for states with no outgoing transitions
            for state in range(num_states):
                if state not in states_with_transitions:
                    builder.add_next_value(state, state, 1.0)

            # Print debugging information
            logging.debug(f"Number of states: {num_states}")
            logging.debug(f"Number of transitions: {len(transitions)}")
            logging.debug(
                f"States with transitions: {sorted(states_with_transitions)}"
            )
            trans_matrix = builder.build()
        return trans_matrix

    def _build_label_func(
        self,
        states: list[VideoState],
        props: list[str],
        model_type: str = "nondeterministic",
    ) -> stormpy.storage.StateLabeling:
        """Build label function.

        Args:
            states (list[State]): List of states.
            props (list[str]): List of propositions.
            model_type (str): Type of model
                ("nondeterministic" or "deterministic").

        Returns:
            stormpy.storage.StateLabeling: State labeling.
        """
        if model_type == "nondeterministic":
            state_labeling = stormpy.storage.StateLabeling(len(states))
            state_labeling.add_label("init")

            for label in props:
                state_labeling.add_label(label)

            for state in states:
                for label in state.descriptive_label:
                    state_labeling.add_label_to_state(label, state.state_index)
        else:
            state_labeling = stormpy.storage.StateLabeling(len(states))

            for prop in props:
                state_labeling.add_label(prop)

            for i, state in enumerate(states):
                for prop in state.props:
                    if prop in props:
                        state_labeling.add_label_to_state(prop, i)
        return state_labeling
