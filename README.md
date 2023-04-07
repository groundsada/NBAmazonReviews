# Naive Bayes Classifier for Sentiment Analysis on Amazon Product Reviews

This project is a part of the CS 481 Artificial Intelligence Language Understanding course at Illinois Institute of Technology. The goal of the project is to implement a Naive Bayes Classifier from scratch to perform sentiment analysis on the Amazon Product Reviews dataset obtained from Kaggle.

The authors of this project are:
- Mohammad Firas Sada (msada@hawk.iit.edu)
- Aleksander Popovic (apopovic@hawk.iit.edu)

## Dataset

The dataset used in this project is the Amazon Product Reviews dataset which is available on Kaggle. The dataset contains approximately 568,454 records with 8 attributes such as product ID, product title, review title, review text, star rating, helpful votes, total votes, and review date. The sentiment labels are derived from the star ratings, where ratings of 4 or 5 are considered as positive, and 1 or 2 are considered as negative. The dataset is preprocessed to remove duplicates and records with missing values.

## Naive Bayes Classifier

The Naive Bayes Classifier is a probabilistic algorithm used for classification tasks. It works on the assumption of independence between the features of a dataset. The algorithm calculates the probability of each feature given a class and the prior probability of each class. Using Bayes' theorem, it calculates the posterior probability of each class given the observed features. The class with the highest posterior probability is considered as the predicted class for the given features.

## Implementation

The Naive Bayes Classifier is implemented using Python programming language. The dataset is loaded into a Pandas DataFrame and preprocessed to extract the necessary features. The text data is preprocessed by removing stop words, punctuations, and converting the text to lowercase. The Naive Bayes Classifier is trained on a subset of the dataset and tested on the remaining subset. The performance of the classifier is evaluated using various metrics such as accuracy, precision, recall, and F1 score.

## License

All source code included is licensed under the [MIT License.](/LICENSE)
