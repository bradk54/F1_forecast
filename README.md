```bash
new-project
========================
Description:

Projection Organization
-------------

|- data
|  |- external          <- data from third parties
|  |- processed         <- cleaned crisp dataset
|  |- raw               <- original immutable data
|  |- train             <- if building ML model the defacto training set
|  |- test              <- if building ML model the defacto test set
|  |- validation        <- if building ML model the defacto validation set
|
|- models               <- trained serialized models 
|
|- notebooks            <- jupyter notebooks. use numbers for ordering
|                           and underscore description. 
|                           e.g. 1_0_data_exploration.ipynb
|- references           <- data dictionaries, manuals, etc
|
|- reports              <- any final analysis using html, pdf, latex etc
|   |- figures          <- generated graphics, plots or tables 
|
|- requirements.txt     <- requirements file for reporducing env. 
|                           'pip freeze > requirements.txt'
|
|-src
|   |- __init__.py      <-makes src a python module
|   |- data             <- fetching, loading and cleaning data
|   |   |- generate_dataset.py             
|   |- features         <- creating feature sets
|   |   |- build_features.py
|   |- models           <- scripts to train and use models
|       |- train.py
|       |- predict.py
** Some folders have .txt to preserve folder structure
```