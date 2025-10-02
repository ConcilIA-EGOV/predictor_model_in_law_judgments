
## How to run this project?

- First, create a python enviroment (python -m venv env)
- Activate it with "source env/bin/activate" 
- Then, install the dependencies by running `pip install -r requirements.txt`
- Type `make` on root directory to receive the running instrutions
- Pick your desired outcome

### Preprocessing Test
- Requires inserting the compatible dataset (as a csv file) int the "input/" folder
    - Under the name "original.csv"
- The specified parameters fro the pipeline will be at `src/util/parameters.py`
- Must be run with `make preprocessing`
- The terminal won't output anything unless there is a error
- In `model/logs/`, there will be more details of the pipeline process:
    - `model/logs/pipeline.json` contains a dictionary with the pipeline parameters and a summary of the preprocessing the data went through
    - `model/logs/data_preparation.txt` contains a detailed step-by-step of everything done to the dataset in proper order.
- In `model/logs/data/`, there will be intermediate datasets of original one    

### Model Creation
- Same requirements as the Preprocessing Test
- The program will be run with `make run`
- After run, the terminal will output the steps of the pipeline and the simplified model performance
- It will generate a model and it's log in the `model/` folder
    - The model itself will be at `model/<model-name>.<model-extension>`
        - Example: `model/DecisionTree.pkl`
    - The main logs will be at `model/<model-name>-log.txt`
        - They concern the model parameters, performance and pipeline modifications
        - Example: `model/DecisionTree-log.txt`
    - The model Explanation will be at `model/shap_beeswarm.png` and `model/shap_global.png`
- They will also generate the same information as the preprocessing test

### Explainability
- Requires running the Model Creation first
- Must be run with `make shap` or `make shap SENT=x` where x is the desired sentence
- If no sentence is provided, it will simply re-do the global explanations
    - The `model/shap_global.png` and `model/shap_beeswarm.png`
    - shap_global is a simple bar plot showcasing how each used feature affected the model results on average
    - shap_beeswarm is more detailed than the bar plot, showcasing both positive and negative impacts on shap values
- If a valid sentence is provided, it will generate a shap explanation for that instance
    - The result will be at `model/shap_sentenca_<sentence-number>.png`
    - The sentences available for explanation can be found in  `model/logs/data/Test.csv` in the first column

### Hyper-parameters fitting
- Has the same requirements as the Preprocessing Test
- Will catch the model hyperparameters in `src/util/param_grids.py`
- And then use the GridSearchCV library to test all hyperparameters combinations for each model type
- Each model will be saved on `model/best__<model-name>.<extension>`
- And the parameters for the same model will be saved on `model/logs/best_parameters__<model-name>.json`
