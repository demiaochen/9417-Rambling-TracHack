# 9417-Rambling-TracHack
COMP9417 TracHack Project by team The Rambling 

## js_eplot.ipynb, data_analysis1.ipynb and data_analysis2.ipynb

The 3 notebooks are used to analyse data and generate graphs. js_eplot.ipynb generates json files needed to plot

## preprocessing.ipynb
This notebook is used to generate preprocessed csv files for preprocessed data.

Running it will generate preprocessedData.csv and preprocessedEval.csv

## model_selection.py 
contains 12 models to be trained and tested on the preprocessed to do model selection 

consists of three parts:
  - Import and split preprocessed data into train, test dataset
  - 12 Models implementations
  - Evaluations on test dataset by accuracy score, F1 score

Usage: after running preprocessing.ipynb, which output the scv files for preprocessed data, run model_selection.py will using preprocessedData.csv to output the result table for models.
  