# Drug Use Prediction Project
### [Russian]()

## Overview
***

This project, conducted as part of the 2nd-year curriculum at HSE SPb, aims to predict the probability of drug use for individuals based on psychological tests, education level, and race. The primary focus is on predicting both past and future drug use likelihood.

### [Presentation](https://docs.google.com/presentation/d/1SRAx6Q4AqNjn7umS8HGrHtlK8aV99tllMibPskqCLgE/edit#slide=id.g2e2fc8ce46e_3_9)

### [Dataset](https://www.kaggle.com/datasets/mexwell/drug-consumption-classification/data)

## Objective
***

The main objective of the project is to predict the probability of drug use for a specific individual based on various psychological and demographic factors.

## Methodology
***

### Model Selection

The CatBoostClassifier from Yandex was chosen for prediction due to its excellent handling of categorical features.

### Model Training

A separate model was trained for each predicted drug. Precision was chosen as the metric, as it is necessary to minimize the number of false positive results.

### Final Accuracy

- **Cannabis**: 0.8906
- **Coke**: 0.6602
- **Ecstasy**: 0.6946
- **LSD**: 0.6371
- **Mushrooms**: 0.6961

## Research Process
***

The entire research process is documented in a notebook. The result of the project was a Telegram bot that conducts a survey and asks users to complete three tests to identify the tendency to use drugs.

## Control Group
***

After surveying the control group, it was found that the majority of people had a high likelihood of using cannabis. This may be due to the fact that 20% of people in the training sample had not used cannabis, which could have led to model overfitting. It may also be related to the perception of cannabis as a light drug and people's openness to such experiences.

## Other Drugs
***

For the remaining drugs, the likelihood of use was above 0.5 if a person had tried two out of four other drugs (excluding cannabis). This is related to the personality type.

## Minimizing False Positives
***

For people in the control group without drug use experience, the model was often accurate, indicating that the task of minimizing false positive results was successfully achieved.


These results and conclusions suggest the success of the approach used and the need for further research and improvement of the model to increase prediction accuracy.

## Setup
***

Install python dependecies

`pip install -r requirements.txt`

Run locally:

`python3 bot.py`