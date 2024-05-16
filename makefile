all:
	@echo "make run          - to run the program"
	@echo "make clean        - to clean the directory"
	@echo "make test         - to run the test cases"
	@echo "make pytorch-test - to run the pytorch test cases"
	@echo "make scikit-test  - to run the scikit test cases"

run:
	@python3 main.py

clean:
	@rm -rf __pycache__

pytorch-test:
	@python3 studies/pytorch.py

scikit-test:
	@python3 studies/scikit.py

test:
	@python3 studies/pytorch.py > logs/log_pytorch.txt
	@python3 studies/scikit.py > logs/log_scikit.txt
