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
ppinat can be executed using the python script `ppinat.py`:

```console
$ python ppinat.py "Average time to payment" --time 6M --log path_to_xes_file.xes

average time to payment

Result:
the average of the duration between the first time instant when <ppinot4py.model.states.RuntimeState object at 0x7fcfa1d5ac70> - AppliesTo.PROCESS and the last time instant when `ACTIVITY` == 'Payment Handled'
case_end
2017-01-31 00:00:00+00:00    7 days 13:18:21.373134
2017-07-31 00:00:00+00:00    9 days 21:04:40.261481
2018-01-31 00:00:00+00:00   11 days 06:30:45.301488
2018-07-31 00:00:00+00:00   10 days 15:57:35.438766
2019-01-31 00:00:00+00:00   12 days 23:54:50.753718
2019-07-31 00:00:00+00:00   91 days 02:25:29.640000
Freq: 6M, Name: data, dtype: timedelta64[ns]
```

For instance, in the previous case, we are computing the PPI "Average time to payment" using the event log available in the file `path_to_xes_file.xes` and a time aggregation of six months, specified using the option `--time 6M`. The possible values for time aggregation are those described (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases)[here].

You can find all the options of the command as follows:

```console
$ python ppinat.py --help

usage: ppinat.py [-h] [-f FILE] [-l LOG] [-c CONFIG] [-t TIME] [-v] [PPI]

positional arguments:
  PPI                   The ppi that is being computed

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  File with a list of PPIs to compute
  -l LOG, --log LOG     Indicates the log you want to use
  -c CONFIG, --config CONFIG
                        The file with the config
  -t TIME, --time TIME  Time grouper used to compute the ppi (e.g. 1M, 6M, 1Y...)
  -v, --verbose         Prints the results of the parsing and matching
```

Instead of specifying the PPI in the command line, a file that contains a list of PPIs can be provided:

```console
$ python ppinat.py -f listppi.txt -l log_file.xes
```

The file `listppi.txt` is just a text file that contains a list of PPIs, one per line:
```txt
average time to payment
percentage of rejected declarations
time between declaration submitted and declaration approved
```

Finally if neither PPI nor file with PPIs is provided, the application will ask the user for input as follows:

```console
$ python ppinat.py -l log_file.xes

Please, write the performance indicator that you want to compute:

```

## Integration into a Python application

If you want to integrate ppinat into your own Python application, the easiest way to do it is by means of class `PPINat` in module `ppinat.computer`. Basically, you just need to:
1) instantiate the class
2) load the context with the log and, optionally, the parsing model and matching weights to be used, and 
3) compute the PPI using, optionally, a time grouper.

```python
from ppinat.computer import PPINat

ppinat = PPINat()
ppinat.load_context('filename.xes', 'specific')
result = ppinat.resolve_compute('Average time to payment', '6M')
```

## Parser
Step 1 is implemented by a parser that performs entity extraction by using a state-of-the-art procedure for token classification.
This procedure works by fine-tuning a transformer language model with our set of defined entity classes, using a linear layer on top of the hidden-states output of the language model. For building this model, we use the script included in `parser_training/transformer.ipynb`.

A challenge here is that the technique we use requires a considerable amount of training data, especially when dealing with such diverse kinds of (potential) input and entities relevant to our work, which we address through data augmentation. To this end, we define textual patterns commonly found in PPI descriptions. The patterns were handcrafted based on the 165 PPIs from our training collection, making sure that a wide variety of different patterns was included for all measure types. The training phrases generation file with all the patterns is `parser_training/phrases.chatito`. Using these patterns, we use [Chatito](https://rodrigopivi.github.io/Chatito/), which is a natural language generator tool to generate distinct training phrases by combining all different alternatives provided for each pattern.

To train the parser, we used the PPIs included in `input/metrics_dataset.json` that belong to the [BPI Challenge 2013 event log](https://data.4tu.nl/ndownloader/files/24033593).


## Evaluation

We have exhaustively tested ppinat using two data sets:

- `input/metrics_dataset-traffic-test.json` that is a testing dataset of the [Traffic Fines Management Process](https://data.4tu.nl/repository/uuid:270fd440-1057-4fb9-89a9-b699b47990f5). The PPIs were gathered during different BPM courses with undergraduate and master students. They were first introduced some basic concepts on process performance measurement and, more specifically, PPIs. After that, the process of managing road trafic fines was described with the support of a BPMN model (`input/TrafficFinesProcessModel.pdf`). Finally, students were required to define 3 PPIs in natural language for that process, by working either in small groups or individually. These courses were taught at Universidad de Sevilla and at an international winter school, so different nationalities, profiles and backgrounds were present among students. The PPIs gathered in this case as well as the result of the preprocessing and subsequent exclusion of some PPIs are collected in the Excel file `input/TrafficFinesDataset.xlsx`.

- `input/metrics_dataset-domesticDeclarations.json` that is a testing dataset of the [domestic declarations event log of the BPI Challenge 2020](https://data.4tu.nl/ndownloader/files/24031811). In this case, the PPIs were collected using an online questionnaire, through which industry and academic users were asked to provide 3 PPI descriptions each, all related to the process to manage reimbursement declarations of domestic travel costs. The online questionnaire can be found here: https://forms.office.com/Pages/ResponsePage.aspx?id=TmhK77WBHEmpjsezG-bEacVp7LJr9DJHpsgYbyyLLxRUQks2REQ2V0FCVzVVQUlTWk1WSDRCSTE3Vy4u and the gathered PPI descriptions together with the rest of the information collected in the online questionnaire are available in the Excel file `input/DomesticDeclarationsDataset.xlsx`.


A summary of the results obtained after the evaluation against `input/metrics_dataset-traffic-test.json` and `input/metrics_dataset-domesticDeclarations.json` can be found at `results/`.

## Replicability of results

You can replicate the results of the evaluation using the Python script `evaluation.py`:

```shell
$ python evaluation.py
```

By default, it executes the PPIs that have been defined in the dataset for the Traffic Fines Management Process. You can find the details at [`input/metrics_dataset-traffic-test.json`](input/metrics_dataset-traffic-test.json). If the event log of the dataset is not available at `input/event_logs`, then it downloads it automatically. The output is a summary of the PPIs that have been correctly identified and those that haven't. It also generates two files with additional information: `results.csv` and `ppi-results.csv`.

To execute a different dataset, you just need to include the path to the JSON file of the dataset as a parameter:

```shell
$ python evaluation.py input/metrics_dataset-domesticDeclarations.json
```

If you want to try with different weights for the features that are considered during matching, you can do so by editing `evaluation.py`. The weights can be easily configured by hand following the examples included there.

