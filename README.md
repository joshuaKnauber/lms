# Language Models

A few project around natural language processing and language models.

## Setup

Create a virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Makemore

The bigram model implementations are following the [makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) series by Andrej Karpathy which covers a few different types of models.

The notebooks are in the `bigram` folder with some additional notes. The data folder contains the list of names used in the project.

## CS50

### Parser

From CS50, parser for given grammar to read in the sentences under cs50/parser/sentences. The functions preprocess and np_chunk as well as the non-terminals were left to implement.

### Questions

An tf-idf implementation to find the most relevant sentences to a given question. The questions are in the cs50/questions/corpus.
The functions left to implement were load_files, tokenize, compute_idfs, tfidf, top_files and top_sentences.
