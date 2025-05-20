# N-gram based next word prediction model

A simple implementation of an N-gram based model with Kneser-Ney Smoothing for predicting the next word in a sequence of text. 

Made this project as part of the NLP course in my University

## Overview

This project implements a next word prediction system using N-gram language modeling. The model analyzes patterns in text sequences to predict the most likely word that follows a given context. It uses the Brown Corpus available in the nltk library as the dataset.

## Requirements

- Python 3.x
- Gradio (for the web interface)

## Project Structure

```
.
├── README.md
├── run.py
└── .gradio/
    └── flagged/
        ├── dataset1.csv
        └── Representative Image/
```

## Usage

To run the prediction model:

```bash
python run.py
```

This will start the Gradio web interface where you can interact with the model.

## Features

- N-gram based text prediction
- Interactive web interface using Gradio
- Dataset logging functionality