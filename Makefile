IMBLEARN_EXPERIMENTS = \
    results/imblearn_tree_low.csv \
    results/imblearn_tree_high.csv \
    results/imblearn_tree_natarajan_low.csv \
    results/imblearn_tree_natarajan_high.csv \
    results/imblearn_tree_natarajan_asymmetric.csv \
    results/imblearn_tree_natarajan_inverse.csv \
    results/imblearn_low.csv \
    results/imblearn_high.csv \
    results/imblearn_natarajan_low.csv \
    results/imblearn_natarajan_high.csv \
    results/imblearn_natarajan_asymmetric.csv \
    results/imblearn_natarajan_inverse.csv
FACT_EXPERIMENTS = \
    results/fact.csv \
    results/fact_fake.csv
DATA = \
    data/crab_precuts.hdf5 \
    data/crab_dl3.hdf5

all: plots experiments

# plot CD diagrams in Julia
plots: results/cdd_f1.tex results/cdd_lima.tex results/cdd_accuracy.tex results/tables.pdf
results/tables.pdf: results/tables.tex results/cdd_f1.tex results/cdd_lima.tex results/cdd_accuracy.tex
	lualatex -interaction=nonstopmode -halt-on-error -output-directory $(dir $@) $<
results/cdd_f1.tex: cdd.jl $(IMBLEARN_EXPERIMENTS) Manifest.toml
	julia --project=. $< --tex $@ --pdf $(patsubst %.tex,%.pdf,$@) --average-table $(patsubst results/cdd_%,results/avg_%,$@) --dataset-table $(patsubst results/cdd_%,results/datasets_%,$@) --metric f1 $(IMBLEARN_EXPERIMENTS)
results/cdd_lima.tex: cdd.jl $(IMBLEARN_EXPERIMENTS) Manifest.toml
	julia --project=. $< --tex $@ --pdf $(patsubst %.tex,%.pdf,$@) --average-table $(patsubst results/cdd_%,results/avg_%,$@) --dataset-table $(patsubst results/cdd_%,results/datasets_%,$@) --metric lima $(IMBLEARN_EXPERIMENTS)
results/cdd_accuracy.tex: cdd.jl $(IMBLEARN_EXPERIMENTS) Manifest.toml
	julia --project=. $< --tex $@ --pdf $(patsubst %.tex,%.pdf,$@) --average-table $(patsubst results/cdd_%,results/avg_%,$@) --dataset-table $(patsubst results/cdd_%,results/datasets_%,$@) --metric accuracy $(IMBLEARN_EXPERIMENTS)
Manifest.toml: Project.toml
	julia --project=. --eval "using Pkg; Pkg.instantiate()"

# one experiment per noise configuration
experiments: $(IMBLEARN_EXPERIMENTS) $(FACT_EXPERIMENTS)
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

results/imblearn_tree_low.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.5 0.1
results/imblearn_tree_high.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.5 0.25
results/imblearn_tree_natarajan_low.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.2 0.2
results/imblearn_tree_natarajan_high.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.4 0.4
results/imblearn_tree_natarajan_asymmetric.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.1 0.3
results/imblearn_tree_natarajan_inverse.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.3 0.1

results/fact.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA)
	venv/bin/python -m pkccn.experiments.fact $@
results/fact_fake.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA)
	venv/bin/python -m pkccn.experiments.fact $@ --fake_labels

# results with confidential FACT data
CONFIDENTIAL_DATA = data/crab_mnoethe_dl2.hdf5 data/crab_mnoethe_dl3.hdf5
confidential: results/confidential_table.tex results/confidential_mrk421.tex results/confidential_mrk501.tex
results/confidential_table.tex: venv/.EXPERIMENTS pkccn/experiments/generate_confidential_table.py results/confidential_crab.csv results/fact.csv
	venv/bin/python -m pkccn.experiments.generate_confidential_table $@ results/fact.csv results/confidential_crab.csv
results/confidential_mrk%.tex: results/confidential_mrk%.csv venv/.EXPERIMENTS pkccn/experiments/generate_mrk_table.py
	venv/bin/python -m pkccn.experiments.generate_mrk_table $@ $<
results/confidential_crab.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA) $(CONFIDENTIAL_DATA)
	venv/bin/python -m pkccn.experiments.fact --dl2_path data/crab_mnoethe_dl2.hdf5 --dl3_path data/crab_mnoethe_dl3.hdf5 --dl2_test_path data/crab_precuts.hdf5 --dl3_test_path data/crab_dl3.hdf5 --no_open_nights $@
results/confidential_mrk421.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA) $(CONFIDENTIAL_DATA) data/mrk421_dl2.hdf5 data/mrk421_dl3.hdf5
	venv/bin/python -m pkccn.experiments.fact --dl2_path data/crab_mnoethe_dl2.hdf5 --dl3_path data/crab_mnoethe_dl3.hdf5 --dl2_test_path data/mrk421_dl2.hdf5 --dl3_test_path data/mrk421_dl3.hdf5 $@
results/confidential_mrk501.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA) $(CONFIDENTIAL_DATA) data/mrk501_dl2.hdf5 data/mrk501_dl3.hdf5
	venv/bin/python -m pkccn.experiments.fact --dl2_path data/crab_mnoethe_dl2.hdf5 --dl3_path data/crab_mnoethe_dl3.hdf5 --dl2_test_path data/mrk501_dl2.hdf5 --dl3_test_path data/mrk501_dl3.hdf5 $@

# test runs of the experiments
results/imblearn_test.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn.py
	venv/bin/python -m pkccn.experiments.imblearn $@ 0.5 0.1 --n_folds 2 --n_repetitions 3 --is_test_run
results/imblearn_tree_test.csv: venv/.EXPERIMENTS pkccn/experiments/imblearn_tree.py
	venv/bin/python -m pkccn.experiments.imblearn_tree $@ 0.5 0.1 --n_folds 2 --n_repetitions 3 --is_test_run
results/fact_test.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA)
	venv/bin/python -m pkccn.experiments.fact $@ --n_repetitions 3 --is_test_run
results/fact_mrk421_test.csv: venv/.EXPERIMENTS pkccn/experiments/fact.py $(DATA)
	venv/bin/python -m pkccn.experiments.fact $@ --n_repetitions 3 --dl2_test_path data/mrk421_dl2.hdf5 --dl3_test_path data/mrk421_dl3.hdf5 --is_test_run

# virtual environment
venv/.EXPERIMENTS: venv/bin/pip setup.py
	venv/bin/pip install --no-binary=tables --no-binary=h5py .[experiments] && touch $@
venv/bin/pip:
	python -m venv venv

.PHONY: plots experiments confidential data
