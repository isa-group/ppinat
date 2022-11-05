class Automaton:
    def __init__(self):
        self.state_map = {}
        self.initial_state = None

    def add_state(self, state_label):
        if state_label not in self.state_map:
            self.state_map[state_label] = State(state_label)

    def set_initial(self, state_label):
        self.initial_state = self.state_map[state_label]

    def increment_transition(self, from_state_label, to_state_label):
        if from_state_label not in self.state_map:
            self.add_state(from_state_label)
        if to_state_label not in self.state_map:
            self.add_state(to_state_label)
        from_state = self.state_map[from_state_label]
        to_state = self.state_map[to_state_label]
        if not from_state.has_outgoing_transition(to_state_label):
            from_state.add_transition(Transition(from_state, to_state))
        from_state.outgoing_transitions_map[to_state_label].increment_count()

    def transition_probability(self, from_state_label, to_state_label):
        if from_state_label not in self.state_map:
            return 0
        transition = self.state_map[from_state_label].outgoing_transitions_map[to_state_label]
        if transition:
            return transition.probability()
        return 0

    def get_outgoing_transitions(self, from_state_label):
        return self.state_map[from_state_label].outgoing_transitions_map.values()

    def total_outgoing_count(self, state_label):
        return self.state_map[state_label].total_outgoing()

    def print(self):
        print(self.state_map.keys())
        for s in self.state_map.values():
            print(s, s.outgoing_transitions_map.values())


class State:
    def __init__(self, label):
        self.label = label
        self.is_accepting = False
        self.outgoing_transitions_map = {}

    def has_outgoing_transition(self, to_state_label):
        return to_state_label in self.outgoing_transitions_map

    def add_transition(self, transition):
        self.outgoing_transitions_map[transition.to_state.label] = transition

    def make_accepting(self):
        self.is_accepting = True

    def total_outgoing(self):
        return sum([t.count for t in self.outgoing_transitions_map.values()])

    def __repr__(self):
        return self.label


class Transition:
    def __init__(self, from_state, to_state):
        self.from_state = from_state
        self.to_state = to_state
        self.count = 0

    def increment_count(self):
        self.count += 1

    def probability(self):
        if self.from_state.total_outgoing() > 0:
            return self.count / self.from_state.total_outgoing()
        return 0

    def __repr__(self):
        return "transition from: " + str(self.from_state) + " to " + str(self.to_state)