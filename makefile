all:
	@echo "make run             - to run the program"
	@echo "make shap (SENT=X)   - to run the SHAP explainability (optionally with a sentence number instead of X)"
	@echo "make clean           - to clear all logs and cache"
	@echo "make parameters      - to run the models parameters test cases"
	@echo "make preprocessing   - to run the preprocessing test cases"

run:
	@rm -rf _logs
	@rm -rf model_*
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
	@rm -rf _logs
	@rm -rf model_*
	@clear