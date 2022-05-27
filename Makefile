# one experiment per noise configuration
experiments: \
    results/imblearn_low.csv \
    results/imblearn_high.csv \
    results/imblearn_natarajan_low.csv \
    results/imblearn_natarajan_high.csv \
    results/imblearn_natarajan_asymmetric.csv \
    results/imblearn_natarajan_inverse.csv
results/imblearn_low.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.5 0.1
results/imblearn_high.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.5 0.25
results/imblearn_natarajan_low.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.2 0.2
results/imblearn_natarajan_high.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.4 0.4
results/imblearn_natarajan_asymmetric.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.1 0.3
results/imblearn_natarajan_inverse.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.3 0.1

# virtual environment
venv/.EXPERIMENTS: venv/bin/pip setup.py
	venv/bin/pip install .[experiments] && touch $@
venv/bin/pip:
	python -m venv venv

.PHONY: experiments
