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

## Running the preprocessing

To run the tokenisation
```bash
!python src/data_processing/loading_tokenization.py
```
This will output three files in the `data/processed` folder: 
`test.pkl`, `train.pkl`, and `tune.pkl`.
Besides, we will also save the tokeniser in `models/tokeniser/tokeniser.pkl`.


## Running the non-peft finetuning

To run the non-parameter efficient finetuning
```bash
!python src/models/finetuning.py
```
This will create a folder in `src/models`: 
`fulldataset_FT_model`, containing the files needed to load in the model and tokeniser.

## Running the LoRA (parameter efficient) finetuning

To run the parameter efficient finetuning, run all cells in this notebook
```bash
!python src/models/LoRA_peft_finetuning.ipynb
```
This will create three folders in `src/models`: 
`lora_trained`, `base_model` and `trained_tokenizer` containing the files needed to load in the base model, the finetuned model and tokeniser.

## Running the streamlit demo

To run the streamlit demo
```bash
streamlit run streamlit_demo.py
```
This will open a local server containing the streamlit demo. For more info visit the streamlit forums on https://docs.streamlit.io/

## Running the Formality Classifier

To tun the Formality Classifier
```bash
!python src/utils/Formality_Classification.py
```
This will import the USE embeddings and then run a cross validation with grid search for the Formality Classifier, while saving the hyper-parameters of the best model

## Running the evaluation of the non-peft model

To tun the evaluation
```bash
!python src/tests/eval_model.py
```

