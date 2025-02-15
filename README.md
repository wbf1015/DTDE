# DTDE: dual teachers for KGE model distillation

## 1. The easiest way to run the code

(1) Check if you have a basic deep learning library with torch installed.

(2) Make sure you have a GPU-accelerated environment with at least 24GB of memory.

(3) Use the homogeneous teacher weights we published for training.

(5) Use the Query-aware negative sampling candidates we published for traning.

(6) Run the following command:

```
1. cd Low_Dimension_KGC
2. bash run.sh
3. bash run1.sh
```

(6) check the metrics and loss in Low_Dimension_KGC/models.



## 2. Start From training teacher models

(1) Training a RotatE teacher model with dim=1024.

(2) Training a LorentzKG teacher model with dim=64.

(3) Both the official implement of RotatE and LorentzKG are published by their author at github.

(4) To construct an homogeneous teacher model, you should first preprocess the dataset by incorporating all inverse relationships, and then identify all potential queries. Subsequently, have the two teacher models independently answer these queries, select the top 100 candidates that are not present in the training set, and document the results. 

(5) And then use Vanilla KD approaches with Query-aware sampling strategy to train homogeneous teacher model.







