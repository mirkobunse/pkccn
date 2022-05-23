# pkccn - Algorithms for learning under partially-known class-conditional label noise

This package implements algorithms for handling class-conditional label noise, also in cases where the class-wise noise rates are (partially) known.

```
pip install git+https://github.com/mirkobunse/pkccn
```

## Usage

To make any soft classifier from [scikit-learn](https://scikit-learn.org/stable/) CCN-aware, you only need to create a `ThresholdedClassifier`.

```python
from pkccn import ThresholdedClassifier

ccn_classifier = ThresholdedClassifier(
    sklearn_base_classifier, # e.g. LogisticRegression
    method, # "menon" or "mithal"
    method_args = {"verbose": True} # optional arguments
)

ccn_classifier.fit(X_train, y_train)
y_pred = ccn_classifier.predict(X_test)
```

## Development / unit testing

During development, run tests with the `unittest` package.

```
python -m venv venv
venv/bin/pip install .[test]
venv/bin/python -m unittest
```
