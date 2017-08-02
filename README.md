Aspect Based Sentiment Analysis
==========================

### Dataset
ABSA-15_Restaurants_Train_Final.xml


### Approach
Natural Language Processing based. Multilabel classifier on top of syntactic extraction rules.



### Results
```bash
[root] [2017-08-02 08:14:54,126] INFO : Dataset: dataset/annoted_data.json
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:18<00:00, 109.20it/s]
[root] [2017-08-02 08:15:12,483] INFO : ================================================================
[root] [2017-08-02 08:15:12,484] INFO : 0.29394702381
[root] [2017-08-02 08:15:12,484] INFO : Data-set Size: 2000
[root] [2017-08-02 08:15:12,484] INFO : Total_aspects Size: 1741
[root] [2017-08-02 08:15:12,484] INFO : Most Efficient Rule: [(1, 483), (3, 275), (11, 274), (-1, 236), (28, 176), (24, 159), (30, 144), (9, 56), (2, 55), (20, 43), (19, 27), (22, 17), (16, 11), (10, 10), (25, 7), (26, 7), (29, 6), (4, 5), (27, 4), (21, 3), (7, 1), (23, 1)]
[root] [2017-08-02 08:15:12,484] INFO : Rules that at least hit one correct: [1, 2, 3, 4, 7, 9, 10, 11, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, -1]
[root] [2017-08-02 08:15:12,487] INFO :
             precision    recall  f1-score   support

          0       0.22      0.23      0.22      1043
          1       0.52      0.51      0.52      1741

avg / total       0.41      0.40      0.41      2784

[root] [2017-08-02 08:15:12,487] INFO : ================================================================
Classification report on training data
              precision    recall  f1-score   support

          0       0.94      0.89      0.91       483
          1       1.00      1.00      1.00        55
          2       0.99      0.99      0.99       275
          3       1.00      1.00      1.00         5
          4       1.00      1.00      1.00         1
          5       1.00      1.00      1.00        56
          6       1.00      0.90      0.95        10
          7       0.98      0.99      0.98       274
          8       1.00      1.00      1.00        11
          9       1.00      0.81      0.90        27
         10       1.00      0.95      0.98        43
         11       1.00      0.67      0.80         3
         12       1.00      0.88      0.94        17
         13       1.00      1.00      1.00         1
         14       0.90      0.70      0.78       159
         15       1.00      1.00      1.00         7
         16       1.00      1.00      1.00         7
         17       1.00      1.00      1.00         4
         18       0.96      0.77      0.85       176
         19       1.00      1.00      1.00         6
         20       0.99      0.90      0.95       144
         21       0.66      0.97      0.78       236

avg / total       0.93      0.91      0.91      2000

[root] [2017-08-02 08:16:24,796] INFO : Dataset: dataset/ABSA15_Restaurants_Test.json
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 582/582 [00:02<00:00, 196.86it/s]
[root] [2017-08-02 08:16:27,759] INFO : ================================================================
[root] [2017-08-02 08:16:27,759] INFO : 0.287334315169
[root] [2017-08-02 08:16:27,759] INFO : Data-set Size: 582
[root] [2017-08-02 08:16:27,759] INFO : Total_aspects Size: 542
[root] [2017-08-02 08:16:27,759] INFO : Most Efficient Rule: [(1, 189), (11, 61), (-1, 61), (3, 55), (30, 53), (28, 52), (24, 40), (2, 19), (9, 12), (20, 10), (19, 7), (10, 6), (16, 3), (25, 3), (26, 3), (22, 2), (27, 2), (21, 2), (7, 1), (29, 1)]
[root] [2017-08-02 08:16:27,759] INFO : Rules that at least hit one correct: [1, 2, 3, 7, 9, 10, 11, 16, 19, 20, 30, 22, 24, 25, 26, 27, 28, 29, -1, 21]
[root] [2017-08-02 08:16:27,760] INFO :
             precision    recall  f1-score   support

          0       0.17      0.25      0.20       246
          1       0.57      0.45      0.51       542

avg / total       0.45      0.39      0.41       788

[root] [2017-08-02 08:16:27,760] INFO : ================================================================
/opt/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
/opt/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1076: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
Classification report on testing_data
              precision    recall  f1-score   support

          0       0.95      0.89      0.92       189
          1       0.95      1.00      0.97        19
          2       0.96      0.98      0.97        55
          3       0.00      0.00      0.00         1
          4       0.00      0.00      0.00        12
          5       0.00      0.00      0.00         6
          6       0.00      0.00      0.00        61
          7       0.00      0.00      0.00         3
          8       0.00      0.00      0.00         7
          9       0.00      0.00      0.00        10
         10       0.00      0.00      0.00        53
         11       0.00      0.00      0.00         2
         12       0.00      0.00      0.00        40
         13       0.00      0.00      0.00         3
         14       0.00      0.00      0.00         3
         15       0.00      0.00      0.00         2
         16       0.00      0.00      0.00        52
         17       0.00      0.00      0.00         1
         18       0.00      0.00      0.00        61
         19       0.00      0.00      0.00         2
         20       0.00      0.00      0.00         0
         21       0.00      0.00      0.00         0

avg / total       0.43      0.42      0.42       582

```


### Paper
Aspect based sentiment analysis for restaurant reviews.


### Setup
```bash
# From project root, execute this command:

pip install -r requirements.txt
```
### Commands

```bash
# From project root, execute this command:
PYTHONPATH='.' python3 main.py
```

#### logs
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
