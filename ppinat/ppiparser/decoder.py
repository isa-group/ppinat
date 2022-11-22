from . import MarkovTrainer as MT
from .markovmodel.HMM import HMM
from .ViterbiDecoder import ViterbiDecoder
import logging
import spacy
import os
import pickle

logger = logging.getLogger(__name__)


def load_decoder(nlp, training_file, parser_serial_file, train_parser=False):
    logger.info("initializing text parser")
    nlp = spacy.load('en_core_web_lg')
    if train_parser or not os.path.exists(parser_serial_file):
        training = MT.read_training_data(training_file)
        hmm = HMM(MT.train_lexicalization_model(training),
                  MT.train_semantic_model(training))
        decoder = ViterbiDecoder(hmm, nlp)
        pickle.dump(hmm, open(parser_serial_file, "wb"))
    else:
        hmm = pickle.load(open(parser_serial_file, "rb"))
        decoder = ViterbiDecoder(hmm, nlp)
    logger.info("Parser initialized")

    return decoder
