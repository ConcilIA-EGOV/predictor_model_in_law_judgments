all:
	@echo "make run            	- to run the program"
	@echo "make shap (SENT=X)   - to run the SHAP explainability (optionally with a sentence number instead of X)"
	@echo "make clean          	- to clear all logs and cache"
	@echo "make parameters  	- to run the models parameters test cases"
	@echo "make preprocessing	- to run the preprocessing test cases"

run:
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf */*/__pycache__
	@rm -rf data/*.csv
	@rm -rf model/logs/*.txt
	@rm -rf model/logs/*.json
	@rm -rf model/logs/*.csv
	@rm -rf model/logs/*/*.csv
	@rm -rf model/*.pkl
	@rm -rf model/*.png
	@rm -rf model/*.pdf
	@rm -rf model/*.txt
	@python main.py

SENT=0
shap:
	@python src/shap_custom.py $(SENT)

preprocessing:
	@python src/formatation/preprocessing.py

parameters:
	@python src/hyperparameters.py

clean:
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf */*/__pycache__
	@rm -rf data/*.csv
	@rm -rf model/logs/*.txt
	@rm -rf model/logs/*.json
	@rm -rf model/logs/*.csv
	@rm -rf model/logs/*/*.csv
	@rm -rf model/*.pkl
	@rm -rf model/*.png
	@rm -rf model/*.pdf
	@rm -rf model/*.txt
