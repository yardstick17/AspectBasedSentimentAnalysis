Aspect Based Sentiment Analysis
==========================

### Dataset
ABSA-15_Restaurants_Train_Final.xml


### Approach
Natural Language Processing based. Multilabel classifier on top of syntactic extraction rules.



### Results
```bash
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:02<00:00, 31.99it/s]
For Data-set:  dataset/annoted_data.json
              precision    recall  f1-score   support

          0       0.17      0.44      0.25       671
          1       0.71      0.40      0.51      2312

avg / total       0.59      0.41      0.45      2983

[root] [2017-08-15 12:35:49,512] INFO : Shape of array for dataset:  X:(2000, 10000) , Y:(2000, 31)
/Users/Amit/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
/Users/Amit/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
Classification report on training data
              precision    recall  f1-score   support

          0       0.00      0.00      0.00         0
          1       0.98      0.99      0.99       406
          2       1.00      1.00      1.00        10
          3       0.99      0.82      0.90       203
          4       1.00      1.00      1.00         3
          5       1.00      1.00      1.00         6
          6       1.00      1.00      1.00        10
          7       1.00      0.97      0.99        39
          8       1.00      1.00      1.00         2
          9       1.00      0.88      0.93         8
         10       1.00      0.83      0.91         6
         11       1.00      0.94      0.97        17
         12       1.00      1.00      1.00         1
         13       0.00      0.00      0.00         0
         14       0.00      0.00      0.00         0
         15       1.00      1.00      1.00         4
         16       0.00      0.00      0.00         0
         17       0.00      0.00      0.00         0
         18       0.00      0.00      0.00         0
         19       0.00      0.00      0.00         0
         20       0.97      0.99      0.98       294
         21       0.00      0.00      0.00         0
         22       0.97      0.99      0.98       103
         23       0.00      0.00      0.00         0
         24       1.00      1.00      1.00         1
         25       0.00      0.00      0.00         0
         26       0.00      0.00      0.00         0
         27       1.00      1.00      1.00        12
         28       0.97      0.99      0.98        87
         29       1.00      1.00      1.00        27
         30       0.00      0.00      0.00         0

avg / total       0.98      0.96      0.97      1239

[root] [2017-08-15 12:36:56,773] INFO : Dataset: dataset/ABSA15_Restaurants_Test.json
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 582/582 [00:12<00:00, 47.23it/s]
For Data-set:  dataset/ABSA15_Restaurants_Test.json
              precision    recall  f1-score   support

          0       0.00      0.00      0.00       105
          1       0.71      0.34      0.46       761

avg / total       0.62      0.30      0.40       866

[root] [2017-08-15 12:37:32,053] INFO : Shape of array for dataset:  X:(582, 10000) , Y:(582, 31)
Classification report on testing_data
              precision    recall  f1-score   support

          0       0.00      0.00      0.00         0
          1       0.99      0.99      0.99       112
          2       1.00      1.00      1.00         2
          3       1.00      0.80      0.89        59
          4       1.00      1.00      1.00         1
          5       1.00      1.00      1.00         3
          6       1.00      1.00      1.00         3
          7       1.00      0.94      0.97        18
          8       1.00      1.00      1.00         1
          9       1.00      1.00      1.00         2
         10       1.00      1.00      1.00         2
         11       1.00      0.88      0.93         8
         12       1.00      1.00      1.00         1
         13       0.00      0.00      0.00         0
         14       0.00      0.00      0.00         0
         15       1.00      1.00      1.00         1
         16       0.00      0.00      0.00         0
         17       0.00      0.00      0.00         0
         18       0.00      0.00      0.00         0
         19       0.00      0.00      0.00         0
         20       0.99      0.99      0.99        72
         21       0.00      0.00      0.00         0
         22       0.97      1.00      0.98        30
         23       0.00      0.00      0.00         0
         24       1.00      1.00      1.00         1
         25       0.00      0.00      0.00         0
         26       0.00      0.00      0.00         0
         27       1.00      1.00      1.00         2
         28       0.96      1.00      0.98        22
         29       1.00      1.00      1.00         6
         30       0.00      0.00      0.00         0

avg / total       0.99      0.95      0.97       346

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
PYTHONPATH='.' python3 training/train_top_classifier.py
```
