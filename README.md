# ppinat: Automated computation of PPIs described in natural language

ppinat is a tool for the computation of PPIs described in natural language. The input is a natural language description of a  PPI and the output is the result of evaluating this PPI against a given event log. The approach consists of four main steps:
1. Step 1 focuses on the extraction of relevant entities from the unstructured PPI description, such as indicators of the measure type, aggregation functions, and the concept to be used to measure the PPI (e.g., activities or data attributes). 
2. Step 2 matches the extracted entities against the contents of the event log in order to start establishing a measurable PPI definition. Then, for cases in which a user left out certain required information, e.g., by not making the start point of a time measure explicit. 
3. Step 3 employs various heuristics to fill in the gaps and thereby complete the PPI definition. 
4. Step 4 uses the established definition in order to compute the desired PPI, thus directly measuring process performance for the event log.

## Installation

To use ppinat, you need to clone this repository and install the dependencies specified in `requirements.txt`. You can do that using the following:

```shell
$ git clone https://github.com/isa-group/ppinat
$ pip3 install -f requirements.txt
```

Note that to execute ppinat you need at least a version of Python >= 3.6


## Execution

You can test ppinat using the Python script `evaluation.py`:

```shell
$ python evaluation.py
```

By default, it executes the PPIs that have been defined in the dataset for the Traffic Fines Management Process. You can find the details at [`input/metrics_dataset-traffic-test.json`](input/metrics_dataset-traffic-test.json). If the event log of the dataset is not available at `input/event_logs`, then it downloads it automatically. The output is a summary of the PPIs that have been correctly identified and those that haven't. It also generates two files with additional information: `results.csv` and `ppi-results.csv`.

If you want to try with different weights for the features that are considered during matching, you can do so by editing `evaluation.py`. The weights can be easily configured by hand following the examples included there.

## Datasets

ppinat comes with three PPI datasets:

- `input/metrics_dataset.json` that was used for training purposes and includes PPIs of the [BPI Challenge 2013 event log](https://data.4tu.nl/ndownloader/files/24033593)
- `input/metrics_dataset-traffic-test.json` that is a testing dataset of the [Traffic Fines Management Process](https://data.4tu.nl/repository/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)
- `input/metrics_dataset-domesticDeclarations.json` that is a testing dataset of the [domestic declarations event log of the BPI Challenge 2020](https://data.4tu.nl/ndownloader/files/24031811)


## Parser

For fine-tuning the language model, we use the script included in `parser_training/transformer.ipynb` and the training phrases generation file `parser_training/phrases.chatito`. To generate the training dataset, you must use [Chatito](https://rodrigopivi.github.io/Chatito/).

The trained model can also be downloaded from  [pytorch_model](https://www.mediafire.com/file/phpx38n1ihc8lcg/pytorch_model.bin/file). It should be included into the `ppinat/PPIBot model` folder. 


