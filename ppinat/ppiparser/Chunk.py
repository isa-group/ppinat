
class Chunk:

    def __init__(self, tag, first_word = None):
        self.tag = tag
        self.words = []
        if first_word:
            self.append_word(first_word)

    def __copy__(self):
        new_instance = Chunk(self.tag)
        new_instance.words.extend(self.words)
        return new_instance

    def text(self):
        return " ".join(self.words)

    def append_word(self, word):
        self.words.append(word)

    def count_sequence_occurrences(self, sequence):
        return sum([self._contains_sequence(sequence, i) for i in range(len(self.words))])

    def _contains_sequence(self, sequence, start_index):
        return sequence == self.words[start_index: start_index + len(sequence)]

    def __repr__(self):
        return " ".join(self.words) + "\\" + self.tag


