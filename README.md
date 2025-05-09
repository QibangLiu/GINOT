# Geometry-Informed Neural Operator Transformer
<img src="images/ginot.png" alt="Overview of model architectures" width="100%">

Figure 1: Overview of GINOT architecture. The boundary points cloud is initially processed through
    sampling and grouping layers to extract local geometric features. These local
    features are then fused with global geometric information via a cross-attention
    layer. This is followed by a series of self-attention layers and a final
    linear layer, producing the KEY and VALUE matrices for the cross-attention layer
    in the solution decoder. In the solution decoder, an MLP encodes the query points
    into the QUERY matrix for the cross-attention layer, which integrates the
    geometry information from the encoder. The output of the cross-attention layer
    is subsequently decoded into solution fields at the query points using another
    MLP.


## Examples
<img src="images/puc_median_sample.gif" alt="Animation of pub median case" width="100%"/>

Figure 2: Visualization of Mises stress and displacement solutions for the median testing case of the micro-periodic unit cell. The first column shows the input surface points cloud, the second column presents the true stress on the actual deformed shape, the third column depicts the predicted stress on the predicted deformed shape, and the fourth column highlights the absolute error of stress on the actual deformed shape.



<img src="images/jeb_plot.png" alt="JEB results" width="100%"/>
Figure 3: Mises stress solutions for the median testing samples of the JEB dataset. The first column shows the input surface points cloud, the second column presents the ground truth from finite element analysis, the third column displays the GINOT prediction, and the last column highlights the absolute error between the prediction and the ground truth.

- [Best Case Visualization](https://QibangLiu.github.io/GINOT/images/test_best.html)
- [Median Case Visualization](https://QibangLiu.github.io/GINOT/images/test_50percentile.html)
- [Worst Case Visualization](https://QibangLiu.github.io/GINOT/images/test_worst.html)

## Dataset
The dataset used for training and evaluation is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.15293036), except for the micro-periodic unit cell dataset, which is avilable on https://doi.org/10.5281/zenodo.15121966. Please download these datasets and unzip into ./data

## Reference
- Liu, Q.; Zhong, V.; Meidani, H.; Abueidda, D.; Koric, S.; Geubelle, P. Geometry-Informed Neural Operator Transformer. arXiv April 29, 2025. https://doi.org/10.48550/arXiv.2504.19452.
