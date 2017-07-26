Aspect Based Sentiment Analysis
==========================

###Dataset
ABSA-15_Restaurants_Train_Final.xml


###Approach
Natural Language Processing based. Multilabel classifier on top of syntactic extraction rules.



###Results
```bash
             precision    recall  f1-score   support

          0       0.85      0.77      0.81      1583
          1       0.72      0.81      0.76      1134

avg / total       0.79      0.79      0.79      2717
```


###Paper
Aspect based sentiment analysis for restaurant reviews.


###Setup
```bash
# From project root, execute this command:

pip install -r requirements.txt
```
###Commands

```bash
# From project root, execute this command:
PYTHONPATH='.' python3 main.py
```

####logs
```bash


100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:18<00:00, 105.86it/s]
[root] [2017-07-26 07:57:56,460] INFO : 0.292283333333
[root] [2017-07-26 07:57:56,460] INFO : Correct Predictions: 0, Empty Correct Predictions : 0, Non empty_miss_case: 0, Data-set Size: 2000
[root] [2017-07-26 07:57:56,460] INFO : Most Efficient Rule: [(-1, 1223), (30, 313), (28, 195), (11, 79), (9, 58), (24, 50), (20, 17), (4, 9), (2, 8), (27, 8), (29, 8), (26, 7), (3, 6), (22, 6), (25, 5), (19, 3), (16, 2), (7, 1), (23, 1), (21, 1)]
[root] [2017-07-26 07:57:56,461] INFO : Rules that at least hit one correct: [2, 3, 4, 7, 9, 11, 16, 19, 20, 30, 22, 23, 24, 25, 26, 27, 28, 29, -1, 21]
[root] [2017-07-26 07:57:56,463] INFO :
             precision    recall  f1-score   support

          0       0.85      0.77      0.81      1583
          1       0.72      0.81      0.76      1134

avg / total       0.79      0.79      0.79      2717
```
