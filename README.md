# Disaster-Tweets-Detection

## A bit about me
🚀 Hi there! I'm Bin Feng, a Business Intelligence Engineer with a burning passion for all things Data Science and Machine Learning. I thrive on the thrill of exploring data, extracting insights, and turning them into actionable strategies.

📊 My journey in this field has been incredible, but I'm always hungry for more knowledge and skills. I firmly believe that continuous learning is the key to staying at the forefront of this dynamic industry. That's why I'm constantly seeking opportunities to sharpen my skills and delve into advanced models.

🤝 Collaboration is at the heart of my work ethic. I'm eager to team up with like-minded individuals to create something truly exceptional. Whether it's a groundbreaking project or a fascinating experiment, I'm all ears for fresh ideas and open to any advice or suggestions that can elevate our work.

💡 Let's innovate, explore, and make a positive impact together. Feel free to reach out, and let's embark on this exciting journey of data-driven discovery!

Thanks for connecting! 🌟

## Objectives
The main objective for this project is to use NLP technics and build deep learning model to identify tweets about disasters using given trianing data. In this project, I used a deep learning model with LSTM (long short-term memory) networks to achieve a high accuracy model.

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
3. If you want to use a pretrained model, you can download it in [My Drive Folder](https://drive.google.com/file/d/10BsvgvPR4TkFi8D8l3babGbWO3vKXK2B/view?usp=drive_link)  
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
![image](https://github.com/bfeng1/Disaster-Tweets-Detection/assets/65517574/36d287b7-f5d5-4bc8-a686-fa7ac39748b6)
![image](https://github.com/bfeng1/Disaster-Tweets-Detection/assets/65517574/3e94f7d1-4651-47ea-9387-1d20ec26bbdd)

### Additional Info
* [Kaggle Project Link](https://www.kaggle.com/code/binfeng2021/nlp-with-disaster-tweets)
* [Author LinkedIn Bin Feng](https://www.linkedin.com/in/bfeng1/)


