EXPERIMENTS = \
    results/imblearn_low.csv \
    results/imblearn_high.csv \
    results/imblearn_natarajan_low.csv \
    results/imblearn_natarajan_high.csv \
    results/imblearn_natarajan_asymmetric.csv \
    results/imblearn_natarajan_inverse.csv

# plot CD diagrams in Julia
plots: results/cdd_f1.tex results/cdd_accuracy.tex
results/cdd_f1.tex: cdd.jl Manifest.toml $(EXPERIMENTS)
	julia --project=. $< --tex $@ --pdf $(patsubst %.tex,%.pdf,$@) --metric f1 $(EXPERIMENTS)
results/cdd_lima.tex: cdd.jl Manifest.toml $(EXPERIMENTS)
	julia --project=. $< --tex $@ --pdf $(patsubst %.tex,%.pdf,$@) --metric lima $(EXPERIMENTS)
results/cdd_accuracy.tex: cdd.jl Manifest.toml $(EXPERIMENTS)
	julia --project=. $< --tex $@ --pdf $(patsubst %.tex,%.pdf,$@) --metric accuracy $(EXPERIMENTS)
Manifest.toml: Project.toml
	julia --project=. --eval "using Pkg; Pkg.instantiate()"

# one experiment per noise configuration
experiments: $(EXPERIMENTS)
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

# test runs of the experiments
results/imblearn_test.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.5 0.1 --n_folds 2 --n_repetitions 1 --is_test_run

# virtual environment
venv/.EXPERIMENTS: venv/bin/pip setup.py
	venv/bin/pip install .[experiments] && touch $@
venv/bin/pip:
	python -m venv venv

.PHONY: plots experiments
