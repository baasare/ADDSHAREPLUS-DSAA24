## AddShare+: Efficient Selective Additive Secret Sharing Approach for Private Federated Learning - DSAA 2024

This repository has all the code used in the experiments carried out in the paper *"AddShare+: Efficient Selective Additive Secret Sharing Approach for Private Federated Learning"* [1].


This repository is organized as follows:

* **./** root - contains all the code for reproducing the experiments described in the paper;
* **helpers** folder - contains helper files and constants used in the main experiment code;
* **resources** folder - contains dataset distribution and encryption keys;


### Requirements

The experimental design was implemented in Python language. Both code and data are in a format suitable for R environment.

In order to replicate these experiments you will need a working installation
  of python. Check [https://www.python.org/downloads/]  if you need to download and install it.

In your python installation you also need to install the following key python packages:

  - Tensorflow
  - Keras
  - Cryptography


  All the above packages, together with other unmentioned dependencies, can be installed via pip. Essentially you need to issue the following command within the code sub-folder:

```r
pip install -r requirements.txt
```

*****

### References
[1] Asare, B. A. and Branco, P. and Kiringa, I. and Yeap, T. (2024) *"AddShare+: Efficient Selective Additive Secret Sharing Approach for Private Federated Learning"*  DSAA 2024 Special Session on Private, Secure, and Trust Data
Analytics (PSTDA), San Diego, USA. (to appear).
