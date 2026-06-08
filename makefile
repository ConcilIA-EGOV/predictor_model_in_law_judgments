all:
	@echo "make run             - to run the program"
	@echo "make shap (SENT=X)   - to run the SHAP explainability (optionally with a sentence number instead of X)"
	@echo "make clean           - to clear all logs and cache"
	@echo "make visualization   - to run the visualization test cases"

run:
	@rm -rf _logs
	@mkdir -p _logs
	@rm -rf _Models*
	@mkdir -p _Models
	@python main.py > _logs/log.txt
	@ cat _logs/log.txt

SENT=0
shap:
	@rm -rf _logs*
	@python src/shap_custom.py $(SENT)

visualization:
	@rm -rf _logs*
	@rm -rf _Separated_Features
	@python src/formatation/visualization.py

clean:
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf */*/__pycache__
	@rm -rf _logs
	@rm -rf _log*
	@rm -rf model_*
	@rm -rf _Model_*
	@clear