from .Automaton import Automaton
from .. import Tags


class SemanticModel(Automaton):

    def get_tags(self):
        return self.state_map.keys()

    def is_accepting(self, tag):
        return self.state_map[tag].has_outgoing_transition(Tags.END_TAG)

    def compute_viterbi_probability(self, viterbi_state):
        prob = 1.0
        tags = viterbi_state.get_tag_sequence()
        for i in range(len(tags) - 1):
            prob = prob * self.transition_probability(tags[i], tags[i + 1])
        return prob
