all:
	@echo "make run              - to run the program"
	@echo "make clean            - to clean the directory"
	@echo "make test             - to run the models paramters test cases"
	@echo "make training-test    - to run and get a log of the model training cases"
	@echo "make test-formatation - to run the formatation test cases"

run:
	@python3 main.py

clean:
	@rm -rf __pycache__

training-test:
	@python3 main.py
# > logs/training_output.txt

test:
	@python3 studies/model_parameters.py > logs/params.txt

test-formatation:
	@ python3 formatation/input_formatation.py > logs/formatation.txt

commit:
	git add .
	git commit -m "Test"
	git push
