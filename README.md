# pyscfit

![test workflow](https://github.com/ogdenkev/pyscfit/actions/workflows/tests.yml/badge.svg)

_Single channel mechanism rate fitting in Python_

**pyscfit** is a Python package for fitting the rates of an ion channel gating mechanism from idealized openings and closings. It uses the theory of Hawkes, Jalali, and Colquhoun to correct for a finite resolution of the open and shut durations.

## Requirements

- Python 3.7+
- [NumPy](https://github.com/numpy/numpy) (>= 1.17.0)
- [SciPy](https://github.com/scipy/scipy) (>= 1.3.0)

## Installation

```console
pip install git+https://github.com/ogdenkev/pyscfit.git@main
```

## Reference Papers

Hawkes, A. G., Jalali, A., & Colquhoun, D. (1990). [The Distributions of the Apparent Open Times and Shut Times in a Single Channel Record when Brief Events Cannot Be Detected](http://www.jstor.org/stable/76804). _Philosophical Transactions: Physical Sciences and Engineering_, 332(1627), 511–538.

Hawkes, A. G., Jalali, A., & Colquhoun, D. (1992). [Asymptotic Distributions of Apparent Open Times and Shut Times in a Single Channel Record Allowing for the Omission of Brief Events](http://www.jstor.org/stable/57135). _Philosophical Transactions: Biological Sciences_, 337(1282), 383–404.

Colquhoun, D., Hawkes, A. G., & Srodzinski, K. (1996). [Joint Distributions of Apparent Open and Shut Times of Single-Ion Channels and Maximum Likelihood Fitting of Mechanisms](http://www.jstor.org/stable/54665). _Philosophical Transactions: Mathematical, Physical and Engineering Sciences_, 354(1718), 2555–2590.
