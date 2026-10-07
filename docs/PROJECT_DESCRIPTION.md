# Project Description

## Problem statement

The project aims to reduce inventory losses from overstocking and prevent lost sales from stockouts by predicting daily retail sales.

## Objective

Predict retail `Units Sold` using historical store/product information, calendar features, categorical variables, and recent sales lags.

## Data processing

The supplied workflow converts dates to datetime, extracts year/month/day-of-week features, sorts by store/product/date, and creates three lag features for previous sales observations.

## Evaluation

A chronological split is used rather than a random split, with the split date set to `2023-01-01`. This is appropriate to the time-ordered nature of the forecasting problem.
