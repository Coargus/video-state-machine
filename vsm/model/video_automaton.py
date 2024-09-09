from cog_cv_abstraction.schema.video_frame import VideoFrame

from vsm.state.video_state import VideoState

from ._base import VideoModel


class VideoAutomaton(VideoModel):
    """Represents a Markov Automaton for video state modeling."""

    def __init__(self, include_initial_state: bool = False) -> None:
        """Initialize the MarkovAutomaton.

        Args:
            include_initial_state (bool, optional): Whether to include
                the initial state. Defaults to False.
            proposition_set (list[str] | None, optional): List of propositions.
                Defaults to None.
        """
        self.previous_states: list[VideoState] = []
        self.states: list[VideoState] = []
        self.transitions = []
        self.include_initial_state = include_initial_state

    def set_up(self, proposition_set: list[str]) -> None:
        """Set up the MarkovAutomaton."""
        self.proposition_set = proposition_set
        self.label_combinations = self._create_label_combinations(
            len(proposition_set)
        )
        self.probability_of_propositions = [
            [] for _ in range(len(proposition_set))
        ]
        self.frame_index_in_automaton = 0

        if self.include_initial_state:
            initial_state = VideoState(
                state_index=0,
                frame_index=-1,
                label="init",
                proposition_set=proposition_set,
            )
            self.previous_states = [initial_state]
            self.states = [initial_state]
            self._current_state = initial_state

    def reset(self) -> None:
        """Reset automaton."""
        self.__init__(self.include_initial_state)
        self.set_up(self.proposition_set)

    def add_frame(self, frame: VideoFrame) -> None:
        """Add frame to automaton."""
        self._get_probability_of_propositions(frame)
        current_states = []
        for prop_comb in self.label_combinations:
            # iterate through all possible combinations of T and F
            self._current_state = VideoState(
                state_index=len(self.states),
                frame_index=self.frame_index_in_automaton,
                label=prop_comb,
                proposition_set=self.proposition_set,
            )
            # TODO: Make a method for update and compute probability
            self._current_state.update(
                frame_index=self.frame_index_in_automaton,
                target_label=prop_comb,
            )
            self._current_state.compute_probability(
                probabilities=self.probability_of_propositions
            )
            if self._current_state.probability > 0:
                self.states.append(self._current_state)
                current_states.append(self._current_state)

        # Build transitions from previous states to current states
        if self.previous_states:
            for prev_state in self.previous_states:
                for cur_state in current_states:
                    transition = (
                        prev_state.state_index,
                        cur_state.state_index,
                        cur_state.probability,
                    )
                    self.transitions.append(transition)
        else:
            # No init state, so add transitions from current states to current states
            for cur_state in current_states:
                transition = (
                    cur_state.state_index,
                    cur_state.state_index,
                    cur_state.probability,
                )
                self.transitions.append(transition)

        self.previous_states = (
            current_states if current_states else self.previous_states
        )
        self.frame_index_in_automaton += 1

    def _get_probability_of_propositions(self, frame: VideoFrame) -> None:
        """Update the probability of propositions."""
        for i, prop in enumerate(self.proposition_set):
            if frame.object_of_interest.get(prop):
                probability = frame.object_of_interest[prop].get_probability()
            else:
                probability = 0.0
            self.probability_of_propositions[i].append(probability)

    def _create_label_combinations(self, num_props: int) -> list[str]:
        """Create all possible combinations of T and F for the number of propositions.

        Args:
            num_props (int): Number of propositions.

        Returns:
            list[str]: List of all possible combinations of T and F.
        """  # noqa: E501
        label_list = []

        def add_labels(
            num_props: int, label: str, label_list: list[str]
        ) -> None:
            if len(label) == num_props:
                label_list.append(label)
                return
            add_labels(num_props, label + "T", label_list)
            add_labels(num_props, label + "F", label_list)

        add_labels(num_props, "", label_list)
        return label_list
