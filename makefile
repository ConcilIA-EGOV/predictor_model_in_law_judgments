all:
	@echo "make run          - to run the program"
	@echo "make clean        - to clean the directory"
	@echo "make test         - to run the models test cases"
	@echo "make training     - to run the model training cases"

run:
	@python3 main.py

clean:
	@rm -rf __pycache__

training:
	@python3 src/training.py

test:
	@python3 studies/model_parameters.py
