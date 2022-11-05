from .Automaton import Automaton
from .. import Tags


class LexicalizationModel(Automaton):

    def __init__(self):
        super().__init__()
        self.tags = set()

    def add_tag_state(self, state_label):
        self.tags.add(state_label)
        self.add_state(state_label)

    def count_word_occurrences(self, word, tag):
        return self.count_subsequence_occurrences([word], tag)

    def count_subsequence_occurrences(self, subsequence, tag):
        count = 0
        for transition in self.get_outgoing_transitions(tag):
            target_sequence = transition.to_state.label.split()
            # if special tag, only check if the subsequence occurs at the start
            if tag in Tags.TYPE_TAGS or tag in Tags.DIVIDER_TAGS:
                new_count = self._contains_subsequence_from_index(
                    target_sequence, subsequence, 0)
                multiplier = transition.count
                count += new_count * multiplier
            else:  # count all occurrences of subsequences
                count += self._count_subsequence_occurrences(
                    target_sequence, subsequence) * transition.count
        return count

    def _count_subsequence_occurrences(self, sequence, subsequence):
        return sum([self._contains_subsequence_from_index(sequence, subsequence, i) for i in range(len(sequence))])

    def _contains_subsequence_from_index(self, sequence, subsequence, start_index):
        return subsequence == sequence[start_index: start_index + len(subsequence)]

    def sequence_equals_reachable_state(self, sequence, tag):
        for transition in self.get_outgoing_transitions(tag):
            if sequence == transition.to_state.label.split():
                return True
        return False
