## Aggregate Tree

Our AggTree is a BallTree variant optimized for high query speeds & reduced memory usage at the sacrifice of accuracy. Our AggTree works by trying to reduce our dataset to a series of aggregate nodes based on proximity. Instead of summing the kernel contributions of significant points we sum a mixture from a smaller set of aggregates (alongside raw data that wasn't aggregated).

The following heatmap was generated using a bandwidth of 0.05 and an atol of 0.001

![KDE Heatmap](kde_heatmap.compare.png)

The above heatmap is a best-case scenario for our AggTree. The dataset was generated using scikit-learn's make blobs with a STD of 0.04, just below our bandwidth.

### Implementation

Tree construction works the exact same as our standard ball tree, but we stop splitting a node when its approximation error is estimated to be below a user-specified absolute tolerance (`atol`). We then calculate the centroid, variance, 3rd & 4th moments of the point-to-centroid distances. We also compute a worst-case error bound for using the Taylor approximation instead of exact evaluation. If this bound falls below `atol`, the node becomes an aggregate leaf and its children are never created.

The error bounds are kernel-dependent. For the Gaussian kernel, we use a 5th-order Taylor remainder:

$$\epsilon \leq \frac{n}{120} \cdot \sup|K^{(5)}| \cdot \frac{R^5}{h^5}$$

For compact-support kernels (Epanechnikov, Uniform, Triangular), the polynomial part of the kernel has exact finite-order derivatives, so the only source of error is points straddling the support boundary. We bound this as:

$$\epsilon \leq n \cdot \frac{R}{h} \cdot K_{\max}$$

Once aggregate nodes are identified, we recurse through the tree and free all data belonging to them, the only values needed to calculate their contribution are the precomputed moments.

For queries, we recurse through the tree pruning nodes that are too far away to make a meaninful contribution. This works the same as a ball tree until we reach an aggregate node. We use a 4th-order Taylor expansion to approximate the aggregate node's contribution:

$$\hat{K} = n \cdot \left( K(r_c) + \frac{1}{2} K''(r_c) \cdot \sigma^2 + \frac{1}{6} K'''(r_c) \cdot \mu_3 + \frac{1}{24} K''''(r_c) \cdot \mu_4 \right)$$

Where $r_c$ is the distance from the query point to the node's centroid, and the moments are:

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^2, \quad \mu_3 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^3, \quad \mu_4 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^4$$