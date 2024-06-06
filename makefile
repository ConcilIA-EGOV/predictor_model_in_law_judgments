all:
	@echo "make run              - to run the program"
	@echo "make clean            - to clean the directory"
	@echo "make test             - to run the models paramters test cases"
	@echo "make training         - to run the model training cases"
	@echo "make test-formatation - to run the formatation test cases"

run:
	@python3 main.py

clean:
	@rm -rf __pycache__

training:
	@python3 src/training.py

test:
	@python3 studies/model_parameters.py

test-formatation:
	@ python3 formatation/input_formatation.py
