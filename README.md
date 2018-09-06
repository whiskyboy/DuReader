# Query-document ranking model

## How to Run

#### Data Schema
The dataset should be stored in JSON format with the schema below: 
```python
{
    segmented_document: [
        [p00, p01, ...],
        ...,
    ],
    segmented_querys: [
        [c00, c01, ...],
        ...,
    ],
    labels (only for train/dev): [
        0, 1, ...,
    ],
}
```

#### Preparation
Before training the model, we have to make sure that the data is ready. For preparation, we will check the data files, make directories and extract a vocabulary for later use. You can run the following command to do this with a specified task name:

```
python run.py --prepare
```
You can specify the files for train/dev/test by setting the `train_files`/`dev_files`/`test_files`. By default, we use the data in `data/demo/`

#### Training
To train the ranking model, you can specify the model type by using `--algo [BIDAF|MLSTM]` and you can also set the hyper-parameters such as the learning rate by using `--learning_rate NUM`. For example, to train a BIDAF model for 10 epochs, you can run:

```
python run.py --train --algo BIDAF --epochs 10
```

The training process includes an evaluation on the dev set after each training epoch. By default, the model with the best accuracy on the dev set will be saved.

#### Evaluation
To conduct a single evaluation on the dev set with the the model already trained, you can run the following command:

```
python run.py --evaluate --algo BIDAF
```

#### Prediction
You can also predict scores for the samples in some files using the following command:

```
python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json 
```

By default, the results are saved at `../data/results/` folder. You can change this by specifying `--result_dir DIR_PATH`.
