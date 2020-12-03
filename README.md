# Amazon_Food_Reviews
In this repository, I am practiced and implemented various machine learning algorithms using the real-world dataset amazon food reviews from Kaggle.

![Amazon_Food_Reviews](https://miro.medium.com/max/523/1*bXDiOoCFTSJJdTQ7JbuijQ.png)

## Task 01. Amazon_Food_Reviews_Featurization

### First We want to know What is Amazon Fine Food Review Analysis?
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review. We also have reviews from all other Amazon categories.

Amazon reviews are often the most publicly visible reviews of consumer products. As a frequent Amazon user, I was interested in examining the structure of a large database of Amazon reviews and visualizing this information so as to be a smarter consumer and reviewer.

Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

### Introduction

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

1. Number of reviews: 568,454
2. Number of users: 256,059
3. Number of products: 74,258
4. Timespan: Oct 1999 — Oct 2012
5. Number of Attributes/Columns in data: 10

### Attribute Information:
1. Id
2. ProductId — unique identifier for the product
3. UserId — unqiue identifier for the user
4. ProfileName
5. Helpfulness Numerator — number of users who found the review helpful
6. HelpfullnessDenominator — number of users who indicated whether they found the review helpful or not
7. Score — rating between 1 and 5
8. Time — timestamp for the review
9. Summary — brief summary of the review
10. Text — text of the review

## Objective
Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).

[Q] How to determine if a review is positive or negative?

[Ans] We could use the Score/Rating. A rating of 4 or 5 could be cosnidered a positive review. A review of 1 or 2 could be considered negative. A review of 3 is nuetral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.

To Know the Complete overview of the Amazon Food review dataset and Featurization visit my medium blog  https://medium.com/analytics-vidhya/amazon-fine-food-reviews-featurization-with-natural-language-processing-a386b0317f56

## Task 02. Apply KNN to Amazon_Food_Reviews_Dataset

![knn](https://miro.medium.com/max/700/0*QPWeWP5FWVMlXNgu.png)

##### Apply Knn versions on these feature sets
- Review text, preprocessed one converted into vectors using (BOW)
- Review text, preprocessed one converted into vectors using (TFIDF)
- Review text, preprocessed one converted into vectors using (AVG W2v)
- Review text, preprocessed one converted into vectors using (TFIDF W2v)


KNN is a non-parametric and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution. In other words, the model structure determined from the dataset. This will be very helpful in practice where most of the real-world datasets do not follow mathematical theoretical assumptions.

KNN is one of the most simple and traditional non-parametric techniques to classify samples. Given an input vector, KNN calculates the approximate distances between the vectors and then assign the points which are not yet labeled to the class of its K-nearest neighbors.

The lazy algorithm means it does not need any training data points for model generation. All training data used in the testing phase. This makes training faster and the testing phase slower and costlier. The costly testing phase means time and memory. In the worst case, KNN needs more time to scan all data points, and scanning all data points will require more memory for storing training data.

To Know how K-NN works visit my medium blog link here https://medium.com/analytics-vidhya/k-nearest-neighbors-algorithm-7952234c69a4.

To Know detailed information about performance metrics used in Machine Learning please visit my medium blog link here https://medium.com/analytics-vidhya/performance-metrics-for-machine-learning-models-80d7666b432e.

After applying KNN algorithm to amazon food reviews data set we make below conclusions.

### Conclusions

![conclusion](https://miro.medium.com/max/569/1*6o-q6t2nK0AiqvwTVWNrWA.png)

1. From the above table, we can conclude the for all the text features best_K by Hyperparameter tuning is 49.
2. From the above table, we observed that the K-NN simple brute model of Word2vec features having the highest AUC score of 84.61% on test data.
3. The K-NN simple brute model of TF-IDF and Bag of words features also works reasonably well on test data having an AUC score of 81.34% and 80.18%.
4. The Avg_Word2Vec and TFIDF_Word2vec are having a low AUC score on test data.

## Task 03. Apply NaiveBayes to Amazon_Food_reviews_Dataset

![Naive_bayes](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLak8IWF3CqnkPQkQ2aMvQ4FOPA5McZsOw_g&usqp=CAU)

### Apply Multinomial NaiveBayes on these feature sets
- Review text, preprocessed one converted into vectors using (BOW)
- Review text, preprocessed one converted into vectors using (TFIDF)

Naive Bayes is a statistical classification technique based on Bayes Theorem. It is one of the simplest supervised learning algorithms. Naive Bayes classifier is a fast, accurate, and reliable algorithm. Naive Bayes classifiers have high accuracy and speed on large datasets.

Naive Bayes is the most straightforward and fast classification algorithm, which is suitable for a large chunk of data. Naive Bayes classifier is successfully used in various applications such as spam filtering, text classification, sentiment analysis, and recommender systems. It uses Bayes theorem of probability for prediction of unknown class.

Naive Bayes makes an assumption that features are conditionally independent. Theoretically, if the assumption does not hold true then the performance of NB degrades. But the research has shown that even if there is some feature dependency the Naive Bayes gives the best result.

To Know detailed information about NaiveBayes algorithm and implementation  please visit my medium blog link here https://medium.com/analytics-vidhya/naive-bayes-algorithm-with-amazon-food-reviews-analysis-66bb59b66e62

After applying Naive Bayes  algorithm to amazon food reviews data set we make below conclusions.

### Conclusions

![conclusion](https://miro.medium.com/max/465/1*g7VGRbn1qsMLtpMxaPzQfQ.png)

1. Compare to Bag of words features representation, TFIDF features are got the highest 95.51% AUC score on Test data.
2. Both are having 0.1 as the best alpha by Hyperparameter tuning.
3. Both models have reasonably worked well for Amazon_food_reviews classification.

