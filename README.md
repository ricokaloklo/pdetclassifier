# pdetclassifier

### Gravitational-wave selection effects using neural-network classifiers

> We present a novel machine-learning approach to estimate  selection biases in gravitational-wave observations. Using techniques similar to those commonly employed in image classification and pattern recognition, we train a series of neural-network classifiers to predict the LIGO/Virgo detectability of gravitational-wave signals from compact-binary mergers. We include the effect of spin precession, higher-order modes, and multiple detectors and show that their omission, as it is common in large population studies, tends to overestimate the inferred merger rate. Although here we train our classifiers using a simple signal-to-noise ratio threshold, our approach is ready to be used in conjunction with full pipeline injections, thus paving the way toward including empirical distributions of  astrophysical and noise triggers into gravitational-wave population analyses.

This is a _re-implementation_ of [`pdetclassifier`](https://github.com/dgerosa/pdetclassifier/tree/master) by Davide Gerosa using PyTorch, instead of Tensorflow (which was limited to Tensorflow 1.x).
