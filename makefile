all:
	@echo "make run             - to run the program"
	@echo "make shap SENT=X     - to run the SHAP explainability (optionally with a sentence number instead of X)"
	@echo "make clean           - to clear all logs and cache"
	@echo "make parameters  	- to run the models parameters test cases"
	@echo "make format 			- to run the formatation test cases"

run:
	@python3 main.py

SENT=0
shap:
	@python3 src/shap_custom.py $(SENT)

parameters:
	@python3 src/hyperparameters.py

format:
	@python3 src/formatation/data_formatation.py

clean:
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf data/*.csv
	@rm -rf model/logs/*.txt
	@rm -rf model/logs/*.json
	@rm -rf model/logs/*.csv
	@rm -rf model/logs/*/*.csv
