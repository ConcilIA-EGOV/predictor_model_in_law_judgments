all:
	@echo "make run             - to run the program"
	@echo "make shap SENT=X     - to run the SHAP explainability (optionally with a sentence number instead of X)"
	@echo "make clean           - to clear all logs and cache"
	@echo "make parameters  	- to run the models parameters test cases"
	@echo "make formatation 	- to run the formatation test cases"

run:
	@python3 main.py > logs/model.txt
	@cat logs/model.txt

SENT=0
shap:
	@python3 src/shap_custom.py $(SENT) > logs/shap.txt
	@cat logs/shap.txt

parameters:
	@python3 src/hyperparameters.py > logs/params.txt
	@cat logs/params.txt

formatation:
	@python3 src/formatation/data_formatation.py > logs/formatation.txt
	@cat logs/formatation.txt

clean:
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf data/*.csv
	@rm -rf logs/*.txt
	@rm -rf logs/*.json
	@rm -rf logs/*.csv
	@rm -rf logs/*/*.csv
	@clear
