# Deepex

Please note that this is research code. It is not optimized, slow, not very 
clean, and not well documented. 

Before proceeding download experiments data available [HERE](https://drive.google.com/file/d/1SCKCgni_iOOJE5ijk7YftvVxtRfaoGBK/view?usp=sharing) and place it into the folder */datasets*. 

#### 1 - Automatic Data Labeling
> python generate_train_data.py *datasets-class* *datasets/generate-datasets*

**datasets/generate-datasets** = a file containing a list with *Class* schema attributes

#### 2 - Train Sentence Classifiers
> python train_classifiers.py *datasets-class*

**datasets-class** = [airline, artist, university, us_county]

#### 3 - Fit parameters and train DL (CNN+BLSTM) model 
> python CNN_BLSTM_fit_hyperparams.py *datasets-class* *property-name*


##### 3.1 - Train BLSTM model
> python BLSTM_train.py *datasets-class* *property-name* *lstm-state-size*


##### 3.2 - Train BLSTM_W2 model
> python BLSTM_w2_train.py *datasets-class* *property-name* *lstm-state-size*


#### 4 - Run Extraction Pipeline
> python deepex_pipeline.py *datasets-class* *dl-network-type* 

**dl-network-type** = [CNN_BLSTM, BLSTM, BLSTM_W2]


# CRF Pipeline

- Requires built training datasets (Step 1)
- Requires trained sentence classifiers (Step 2)

> python crf_pipeline.py *datasets-class*
