# LLM-Group-04
This repository is exclusively for the Large Language Models course from RUG (WBAI068-05, 2024-2025), which will be a informal-to-formal style transfer task

# How to run the project
Create a python package manager (python environment) by typing 
`python -m venv <directory>`
in your terminal, where directory is the name of the folder you want to give to your package manager. A popular option is to call it "venv".

To activate, type 
`source <directory>/bin/activate`
if in linux or macOS or 
```shell
# In cmd.exe
venv\Scripts\activate.bat
# In PowerShell
venv\Scripts\Activate.ps1
```
if in windows.

To install all the requirements type 
`pip install -r requirements.txt`.
However, for the usage of LoRA you will need to install the package from the source. To do so, type
```bash
!pip install -q git+https://github.com/huggingface/peft
```

## Running the sections

To run the tokenisation
```bash
!python src/data_processing/data_preprocessing.py
```
This step should fill the folder data/processed with three files: `test_dataset_processed.csv`, `train_dataset_processed.csv` and `val_dataset_processed.csv`.

There are two types of tuning that can be done: hyperparameter tuning and parameter-efficient hyperparameter tuning.
To run the LoRA model you need to be at the root of the repository and run the following command in terminal (not IDE):
```bash
!python src/utils/LoRA.py
```
This should make a folder `logs` with the results of the training.