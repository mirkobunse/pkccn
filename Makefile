# experiments
experiments: results/imblearn_yeast_natarajanHigh.csv

results/imblearn_yeast_natarajanHigh.csv: venv/.EXPERIMENTS
	venv/bin/python -m pkccn.experiments.imblearn $@ yeast_ml8 0.4 0.4

# virtual environment
venv/.EXPERIMENTS: venv/bin/pip
	venv/bin/pip install .[experiments] && touch $@
venv/bin/pip:
	python -m venv venv

.PHONY: experiments
