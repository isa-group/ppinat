import spacy
import pytest
from helpers import load_log
from matcher.similarity import SimilarityComputer, SlotPair, Slot, choose_best_similarity

@pytest.fixture
def nlp():
    return spacy.load('en_core_web_lg')

@pytest.fixture
def log_bpi2013():
    bpi2013 = 'input/event_logs/bpi_challenge_2013_incidents.xes'
    return load_log(bpi2013)

@pytest.fixture
def log_its():
    its = 'input/event_logs/output_2M_english.csv'
    return load_log(its, id_case="ID", time_column="DATE")

def test_slot_match(log_bpi2013, nlp):
    similarity = SimilarityComputer(log_bpi2013, nlp)

    candidates = similarity.find_most_similar_slot('to resolve an incident')
    slot = choose_best_similarity(candidates)

    assert slot.column1 == 'lifecycle:transition'
    assert slot.value1 == 'Resolved'


def test_column_candidates(log_bpi2013, nlp):
    similarity = SimilarityComputer(log_bpi2013, nlp)
    
    columns = similarity.column_candidates()
    print(columns)

    assert len(columns) == 5

def test_column_candidates_its(log_its, nlp):
    similarity = SimilarityComputer(log_its, nlp)
    
    columns = similarity.column_candidates()
    print(columns)

    assert len(columns) == 4

def test_slot_pair_repr():
    pair = SlotPair(Slot('c1', 'v1'), Slot('c1','v2'))
    repr_pair = str(pair)

    assert repr_pair == "Slot: c1: v1, Slot: c1: v2"

def test_find_most_similar_attribute(log_bpi2013, nlp):
    similarity = SimilarityComputer(log_bpi2013, nlp)
    most_similar = choose_best_similarity(similarity.find_most_similar_attribute('concept', filter_att='slot'))
    assert most_similar == "concept:name"
