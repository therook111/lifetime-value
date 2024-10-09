# TOPIC: Customer Lifetime Value Prediction utilizing Deep Neural Network

### Background:
Customer Lifetime Value (CLV) is known as an important concept in marketing and management of organizations to increase the captured profitability. Total value that a customer produces during his/her lifetime is named customer lifetime value. Accurate predictions of customers' future lifetime value (LTV) given their attributes and past purchase behavior enables a more customer-centric marketing strategy. Marketers can segment customers into various buckets based on the predicted LTV and, in turn, customize marketing messages or advertising copies to serve customers in these different segments better. 

Disclaimer: This repository is essentially a PyTorch implementation of the Wang, Xiaojing, Liu, Tianqi, and Miao, Jingang. (2019). [A Deep Probabilistic Model for Customer Lifetime Value Prediction ](https://arxiv.org/abs/1912.07753)

We simply made the codebase modular for a more OOP-approach and user-friendly as a learning oppurtunity.

**Algorithm used**: Deep Neural Network


### Procedure
- Preprocessing the dataset
- Deciding on the loss function and the model architecture
- Train the model
- Inference

### Dataset
- We have selected the [Kaggle Acquire Valued Shoppers Challenge Dataset](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) as our dataset. The dataset has 350 million rows of anonymous transactions from 300k shoppers.
- The preprocessing of the dataset is as follows:
  + We only take entries with positive purchase amounts as negative indicates product returns.
  + We transformed the dataset to be from transactions-centric to customer-centric, since we're modelling the customer's LTV.
  + Included a column for calibration value - The total value of the customer's first day purchases
  + Included a column for holdout value - The total value of the customer's 1 year period after the first day. The paper mentioned that performance deterioriates after the one-year mark, so we decided 1-year as our cutoff. **This is the label.**
  + Applied log transformation on the calibration value to align with the loss function.
- We then perform these steps on specific companies in the top 20 companies with the largest amount of transactions and we're only going to model the LTV for these companies.
    


### How to use?

- First-off, make sure you have your Kaggle API in the form of `kaggle.json` in your working directory. If you don't have, visit [this site](https://www.kaggle.com/settings/account) and scroll down to find "Create New Token"

- Install the necessary libraries by using
  ```
  pip install -r requirements.txt
  ```

- In your terminal, run
  ```
  !python src/utils/download_data.py
  ```
  This will download the data, and make necessary preparations for the following steps.

- Follow the steps in our `example_usage.ipynb` Jupyter notebook to train the model.


