This directory contains source code of the project for two pipelines **Baseline**(`sklearn`) and **RNN/Transformers**(`torch`). Here is the details of this directory

#### Prepare Dataset
`prepare_dataset.py` contains the function which will create the resulting dataset for the training. It can be run with this command
```
python prepare_dataset.py -d path/to/raw/data/csv -o path/to/output/directory
```
In the output directory 3 splits will be created of the dataset (train | valid | test). In addition to that, config file will be generated containing additional info related to the torch training pipeline. Moreover `label_encoder` for each categorical feature will be saved and also the `scaler` as well, in order to be able to perform the same steps in the upcoming new dataset.

#### Shared State
Dataclass(struct like) containing the fields in order to pass the content from models to datasets or among other objects.


### PyTorch
#### Dataset
`dataset.py` is the abstract class which derives from `torch.utils.data.Dataset`, currently it is assumed that data input has to be csv file and the corresponding config files containing the information about the data, such as the categorical feature names, targer column name, etc.
Within `datasets/` directory there are specific classes which derive and override `__getitem__` method, that is done because each model can have its specific anticipated inputs, such as **Transformers** which expects 2 `torch.Tensor`-s as an input (x_categorical, x_continoius) (see `datasets/transformer_data_loader.py`).
This class can be utilized for any DNN based models which are implemented with torch.

#### Models
`models` directory contains all model types in the project. The `baseline` directory contains whole the necessary codebase for training baseline models utilizing sklearn, etc. `transformer` contains three models implementations and the configs files in order to be integrated to the pipeline and be trained.
`BaseModel` anticipates the necessary config file and shared information from the `dataset`. The other parts of the models implementations are simple, they should only contain `forward` method properly defined.

#### Factory
`Factory` class in the `factory.py` receives the general config file containing whole necessary info about the datasets, model, losses etc. It has methods in order to get the class from the module and instansiate the class object and return it for the later usage.

#### Trainer
`trainer.py` contains the class in order to do the training. It can be initialized with providing the model, datasets(train\valid), loss, optimizer, etc. Call the `train(num_epochs)` method in order to do the training, where class has the protected method for training for single epoch and then performs validation step. Training step is vanilla and simple, forward -> loss -> update weights. In order to extend the procedure new subclass can be created and derived from this class and it should have the `_train_step` function overriden.

* Training
In order to train the model with torch pipeline the `train.py` should be invoked
```
python src/train.py -c configs/transformer_configs.yaml
```
Note that in base directory `results` directory will be created and it will have the subdirectory with the name taken from the configs `output_dirpath` field as the prefix and current date time as postfix. In resulting directory plots of the losses, best model checkpoints will be saved.
* Evaluation
Evaluation on test data from the newly trained model can be done with this command
```
python src/evaluate.py -c path/to/configs/in/trained/model/directory
```
