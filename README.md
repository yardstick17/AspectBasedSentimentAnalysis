Aspect Based Sentiment Analysis
==========================

### Paper

<embed src="https://github.com/yardstick17/AspectBasedSentimentAnalysis/raw/master/review_highlight_paper.pdf" width="700" height="1000" 
 type="application/pdf">

### Dataset
ABSA-15_Restaurants_Train_Final.xml


### Approach
Natural Language Processing based. Multilabel classifier on top of syntactic extraction rules.



### Results
```bash
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 582/582 [02:18<00:00,  3.10it/s]
For Data-set:  dataset/ABSA15_Restaurants_Test.json
              precision    recall  f1-score   support

          0       0.00      0.00      0.00        54
          1       0.83      0.50      0.63       542

avg / total       0.76      0.46      0.57       596

[root] [2017-09-27 17:59:03,966] INFO : Shape of array for dataset:  X:(582, 13635) , Y:(582, 31)
/Users/Amit/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
Classification report on testing_data
              precision    recall  f1-score   support

          0       0.97      0.93      0.95       101
          1       1.00      1.00      1.00         2
          2       1.00      0.54      0.70        54
          3       1.00      1.00      1.00         2
          4       1.00      1.00      1.00         3
          5       1.00      1.00      1.00         2
          6       1.00      0.74      0.85        23
          7       1.00      1.00      1.00         1
          8       0.00      0.00      0.00         1
          9       1.00      0.83      0.91         6
         10       1.00      0.33      0.50         3
         11       1.00      1.00      1.00         1
         12       1.00      1.00      1.00        14
         13       0.94      0.79      0.86        19
         14       1.00      1.00      1.00         1
         15       0.00      0.00      0.00         1
         16       1.00      1.00      1.00         2
         17       1.00      1.00      1.00         5

avg / total       0.97      0.80      0.87       241

[root] [2017-09-27 17:59:19,624] INFO : Dataset: dataset/ABSA15_Restaurants_Test.json
582it [00:00, 628.15it/s]
NO PREDICTION FOR RULE:  397  out of:  582
Task: ONLY_ASPECT_PREDICTION False
Accuracy:  80.96885813148789
Total:  289 , Correct:  234
::::::::::::::::::   TESTING   ::::::::::::::::::
 dataset/ABSA15_Restaurants_Test.json
              precision    recall  f1-score   support

          0       0.00      0.00      0.00        55
          1       0.81      0.43      0.56       542

avg / total       0.74      0.39      0.51       597

```

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
