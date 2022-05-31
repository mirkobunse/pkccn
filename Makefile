EXPERIMENTS = \
    results/imblearn_low.csv \
    results/imblearn_high.csv \
    results/imblearn_natarajan_low.csv \
    results/imblearn_natarajan_high.csv \
    results/imblearn_natarajan_asymmetric.csv \
    results/imblearn_natarajan_inverse.csv

# plot CD diagrams in Julia
plots: results/cdd_f1.tex results/cdd_lima.tex results/cdd_accuracy.tex
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

# inspection of method behavior
results/inspect_objectives_main.pdf: results/inspect_objectives_main.tex \
        results/inspect_objectives_coil_2000.csv \
		results/inspect_objectives_satimage.csv \
		results/inspect_objectives_letter_img.csv \
		results/inspect_objectives_pen_digits.csv \
		results/inspect_objectives_protein_homo.csv \
		results/inspect_objectives_optical_digits.csv \
		results/inspect_objectives_thyroid_sick.csv \
		results/inspect_objectives_sick_euthyroid.csv \
		results/inspect_objectives_mammography.csv \
		results/inspect_objectives_abalone.csv \
		results/inspect_objectives_wine_quality.csv \
		results/inspect_objectives_us_crime.csv \
		results/inspect_objectives_car_eval_34.csv \
		results/inspect_objectives_webpage.csv \
		results/inspect_objectives_isolet.csv \
		results/inspect_objectives_yeast_ml8.csv \
		results/inspect_objectives_scene.csv
	lualatex -interaction=nonstopmode -halt-on-error -output-directory $(dir $@) $<
results/inspect_objectives_coil_2000.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 coil_2000
results/inspect_objectives_satimage.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 satimage
results/inspect_objectives_letter_img.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 letter_img
results/inspect_objectives_pen_digits.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 pen_digits
results/inspect_objectives_protein_homo.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 protein_homo
results/inspect_objectives_optical_digits.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 optical_digits
results/inspect_objectives_thyroid_sick.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 thyroid_sick
results/inspect_objectives_sick_euthyroid.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 sick_euthyroid
results/inspect_objectives_mammography.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 mammography
results/inspect_objectives_abalone.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 abalone
results/inspect_objectives_wine_quality.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 wine_quality
results/inspect_objectives_us_crime.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 us_crime
results/inspect_objectives_car_eval_34.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 car_eval_34
results/inspect_objectives_webpage.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 webpage
results/inspect_objectives_isolet.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 isolet
results/inspect_objectives_yeast_ml8.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 yeast_ml8
results/inspect_objectives_scene.csv: venv/.EXPERIMENTS pkccn/experiments/inspect_objectives.py
	venv/bin/python -m pkccn.experiments.inspect_objectives $@ 0.5 0.1 scene

# virtual environment
venv/.EXPERIMENTS: venv/bin/pip setup.py
	venv/bin/pip install .[experiments] && touch $@
venv/bin/pip:
	python -m venv venv

.PHONY: plots experiments
