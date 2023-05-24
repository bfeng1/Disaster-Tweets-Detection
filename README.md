# Disaster-Tweets-Detection

## Objectives
The main objective for this project is to use NLP technics and build deep learning model to identify tweets about disasters using given trianing data. 

## Data

### [Natural Language Procesing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)
* train.csv - the training set
* test.csv - the test set

#### Acknowledgments
This dataset was created by the company figure-eight and originally shared on their [‘Data For Everyone’ website here.]()

Tweet source: (https://twitter.com/AnyOtherAnnaK/status/629195955506708480)

### [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation)

#### Acknowledgements
This data has been released under the Open Data Commons Public Domain Dedication and License.

```Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. URL: https://nlp.stanford.edu/pubs/glove.pdf```

## Get Started

### Prerequirements

1. Install all needed libraries using ```pip install -r requirements.txt```
2. Download [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation)
3. If you want to use a pretrained model, you can download it in [My Drive Folder](https://drive.google.com/file/d/1THGVUlYhKefu7PYCtCidepm75Pm8aRCL/view?usp=sharing)  
4. Make sure you structure the project folder as following:
```
.
├── Data
│   ├── glove-global-vectors-for-word-representation
│   │   └── glove.6B.200d.txt
│   └── nlp-getting-started
│       ├── test.csv
│       └── train.csv
├── my_model
│   ├── assets
│   ├── fingerprint.pb
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── my_model.zip
├── requirements.txt
└── src
    └── main.py
```

### Use Case

1. In the terminal, run ```python src/main.py```
2. Follow the instruction on the terminal, give user inputs
3. Please note, retrain the model might take a while. If you just want to try it out, you can use the pretrained model instead.


### Model Performance

![image](https://github.com/bfeng1/Disaster-Tweets-Detection/assets/65517574/0278fd02-112d-4203-b304-3a22106308ba)

