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

You also need to download the [pytorch pre-trained model for the parser](https://www.mediafire.com/file/phpx38n1ihc8lcg/pytorch_model.bin/file) (it is not included because of its size).  It should be included into the `ppinat/models/GeneralClassifier` folder. 


## Execution

You can test ppinat using the Python script `evaluation.py`:

```shell
$ python evaluation.py
```

By default, it executes the PPIs that have been defined in the dataset for the Traffic Fines Management Process. You can find the details at [`input/metrics_dataset-traffic-test.json`](input/metrics_dataset-traffic-test.json). If the event log of the dataset is not available at `input/event_logs`, then it downloads it automatically. The output is a summary of the PPIs that have been correctly identified and those that haven't. It also generates two files with additional information: `results.csv` and `ppi-results.csv`.

To execute a different dataset, you just need to include the path to the JSON file of the dataset as a parameter:

```shell
$ python evaluation.py input/metrics_dataset-domesticDeclarations.json
```

If you want to try with different weights for the features that are considered during matching, you can do so by editing `evaluation.py`. The weights can be easily configured by hand following the examples included there.

## Datasets

ppinat comes with three PPI datasets:

- `input/metrics_dataset.json` that was used for training purposes and includes PPIs of the [BPI Challenge 2013 event log](https://data.4tu.nl/ndownloader/files/24033593)
- `input/metrics_dataset-traffic-test.json` that is a testing dataset of the [Traffic Fines Management Process](https://data.4tu.nl/repository/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)
- `input/metrics_dataset-domesticDeclarations.json` that is a testing dataset of the [domestic declarations event log of the BPI Challenge 2020](https://data.4tu.nl/ndownloader/files/24031811)


## Parser

Step 1 is implemented by a parser that performs entity extraction by using a state-of-the-art procedure for token classification.
This procedure works by fine-tuning a transformer language model with our set of defined entity classes, using a linear layer on top of the hidden-states output of the language model. For building this model, we use the script included in `parser_training/transformer.ipynb`.

A challenge here is that the technique we use requires a considerable amount of training data, especially when dealing with such diverse kinds of (potential) input and entities relevant to our work, which we address through data augmentation. To this end, we define textual patterns commonly found in PPI descriptions. The patterns were handcrafted based on the 165 PPIs from our training collection, making sure that a wide variety of different patterns was included for all measure types. The training phrases generation file with all the patterns is `parser_training/phrases.chatito`. Using these patterns, we use [Chatito](https://rodrigopivi.github.io/Chatito/), which is a natural language generator tool to generate distinct training phrases by combining all different alternatives provided for each pattern.

## Test Collection
The test collection consists of two data sets: 
- Its first one was gathered during different BPM courses with undergraduate and master students. They were first introduced some basic concepts on process performance measurement and, more specifically, PPIs. After that, the process of managing road trafic fines was described with the support of a BPMN model (`input/TrafficFinesProcessModel.pdf`). Finally, students were required to define 3 PPIs in natural language for that process, by working either in small groups or individually. These courses were taught at Universidad de Sevilla and at an international winter school, so different nationalities, profiles and backgrounds were present among students. The PPIs gathered in this case as well as the result of the preprocessing and subsequent exclusion of some PPIs are collected in the Excel file `input/TrafficFinesDataset.xlsx`.
- The second one was collected using an online questionnaire, through which industry and academic users were asked to provide 3 PPI descriptions each, all related to the process to manage reimbursement declarations of domestic travel costs. The online questionnaire can be found here: https://forms.office.com/Pages/ResponsePage.aspx?id=TmhK77WBHEmpjsezG-bEacVp7LJr9DJHpsgYbyyLLxRUQks2REQ2V0FCVzVVQUlTWk1WSDRCSTE3Vy4u and the gathered PPI descriptions together with the rest of the information collected in the online questionnaire are available in the Excel file `input/DomesticDeclarationsDataset.xlsx`.
