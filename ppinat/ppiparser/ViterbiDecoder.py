from . import Tags
from .Chunk import Chunk
from .ppiannotation import ViterbiState, PPIAnnotation
import logging

logger = logging.getLogger(__name__)


class ViterbiDecoder:

    def __init__(self, hmm, nlp):
        self.sem_model = hmm.semantic_model
        self.lex_model = hmm.lex_model
        self.nlp = nlp

    def predict_annotation(self, input_string, less_restricted=False) -> PPIAnnotation:
        words = input_string.split()
        logger.info(f"Annotating: {words}")
        paths = self._compute_initial_paths(words[0])

        for word in words[1:]:
            new_paths = []
            # compute normal next paths
            for path in paths:
                if not less_restricted:
                    new_paths.extend(self._compute_next_paths(path, word))
                else:
                    new_paths.extend(
                        self.__compute_next_paths_less_restricted(path, word))
            #  compute paths by skipping a zero prob step
            if not new_paths:
                for path in paths:
                    new_paths.extend(
                        self._compute_next_paths(path, word, skip=True))
            # compute paths based on semantics
            if not new_paths:
                for path in paths:
                    new_paths.extend(self._compute_semantic_paths(path, word))
            paths = new_paths
        # filter out all paths with zero probability
        paths = [
            path for path in paths if self.sem_model.compute_viterbi_probability(path) > 0]
        # if there are no valid paths, use less restrictive method
        if not paths and not less_restricted:
            return self.predict_annotation(input_string, less_restricted=True)
        return self._most_probable_path(paths)

    def _most_probable_path(self, paths):
        best_path = None
        # check for most probable accepting path
        for path in paths:
            if self.sem_model.is_accepting(path.get_last_chunk().tag) and \
                    (not best_path or path.probability > best_path.probability):
                best_path = path
        # check for most probable path
        if not best_path:
            for path in paths:
                if not best_path or path.probability > best_path.probability:
                    best_path = path
        best_path.close_annotation()
        return best_path

    def _compute_initial_paths(self, first_word):
        initial_paths = []
        for transition in self.sem_model.get_outgoing_transitions(Tags.START_TAG):
            tag = transition.to_state.label
            outgoing_count = self.lex_model.total_outgoing_count(tag)
            if outgoing_count > 0:
                trans_prob = self.sem_model.transition_probability(
                    Tags.START_TAG, tag)
                cwk = self.lex_model.count_word_occurrences(first_word, tag)
                chunk = Chunk(tag, first_word)
                state_prob = trans_prob * cwk / outgoing_count
                if state_prob > 0:
                    state = ViterbiState(cwk, state_prob, chunk)
                    initial_paths.append(state)
        if not initial_paths:
            return [self._create_semantic_initial_path(first_word)]
        return initial_paths

    def _compute_next_paths(self, current_state, word, skip=False):
        new_paths = []
        generate_next_tags = True
        last_chunk = current_state.get_last_chunk()
        if last_chunk.tag in Tags.DIVIDER_TAGS:
            generate_next_tags = self.lex_model.sequence_equals_reachable_state(
                last_chunk.words, last_chunk.tag)
        if generate_next_tags:
            for transition in self.sem_model.get_outgoing_transitions(last_chunk.tag):
                target_tag = transition.to_state.label
                p1 = transition.probability()
                cwk = self.lex_model.count_word_occurrences(
                    word, target_tag) if not skip else 1
                outgoing = self.lex_model.total_outgoing_count(target_tag)
                new_chunk = Chunk(target_tag, word)
                prob = p1 * cwk / outgoing
                if prob > 0:
                    new_paths.append(
                        current_state.copy_and_extend(prob, new_chunk))
        # add state that continues with current tag
        new_chunk = last_chunk.__copy__()
        new_chunk.append_word(word)
        cwk = self.lex_model.count_subsequence_occurrences(
            new_chunk.words, last_chunk.tag) if not skip else 1
        prob = cwk * 1.0 / current_state.cwk
        if prob > 0.0:
            current_state.remove_last_chunk()
            new_paths.append(current_state.copy_and_extend(prob, new_chunk))
        return new_paths

    def _compute_semantic_paths(self, current_state, word):
        new_paths = []
        word_vec = self.nlp(word)
        for type_tag in Tags.TYPE_TAGS:
            type_vec = self.nlp(type_tag)
            sim_score = word_vec.similarity(type_vec)
            cwk = 1
            outgoing = self.lex_model.total_outgoing_count(type_tag)
            new_chunk = Chunk(type_tag, word)
            prob = sim_score * cwk / outgoing
            if prob > 0:
                new_paths.append(
                    current_state.copy_and_extend(prob, new_chunk))
        return new_paths

    def _create_semantic_initial_path(self, first_word):
        word_vec = self.nlp(first_word)
        best_path = None
        for type_tag in Tags.TYPE_TAGS:
            type_vec = self.nlp(type_tag)
            sim_score = word_vec.similarity(type_vec)
            if sim_score > 0:
                chunk = Chunk(type_tag, first_word)
                state = ViterbiState(1, sim_score, chunk)
                if not best_path or state.probability > best_path.probability:
                    best_path = state
        if not best_path:
            chunk = Chunk(Tags.START_TAG, first_word)
            best_path = ViterbiState(1, 1, chunk)
            best_path.chunks = []
            best_path.chunks.append(chunk)
        return best_path

    def _compute_next_paths_less_restricted(self, current_state, word, skip=False):
        new_paths = []
        generate_next_tags = True
        last_chunk = current_state.get_last_chunk()
        if last_chunk.tag in Tags.DIVIDER_TAGS:
            generate_next_tags = self.lex_model.sequence_equals_reachable_state(
                last_chunk.words, last_chunk.tag)
        if generate_next_tags:
            for transition in self.sem_model.get_outgoing_transitions(last_chunk.tag):
                target_tag = transition.to_state.label
                p1 = transition.probability()
                cwk = self.lex_model.count_word_occurrences(
                    word, target_tag) if not skip else 1
                outgoing = self.lex_model.total_outgoing_count(target_tag)
                new_chunk = Chunk(target_tag, word)
                prob = p1 * cwk / outgoing
                if prob > 0:
                    new_paths.append(
                        current_state.copy_and_extend(prob, new_chunk))
        # add state that continues with current tag
        new_chunk = last_chunk.__copy__()
        new_chunk.append_word(word)
        cwk = self.lex_model.count_subsequence_occurrences(
            new_chunk.words, last_chunk.tag) if not skip else 1
        prob = cwk * 1.0 / current_state.cwk
        if prob > 0.0:
            current_state.remove_last_chunk()
            new_paths.append(current_state.copy_and_extend(prob, new_chunk))
        if last_chunk.tag in Tags.EVENT_TAGS:
            return [current_state.copy_and_extend(1 / 9999999, new_chunk)]
        return []
