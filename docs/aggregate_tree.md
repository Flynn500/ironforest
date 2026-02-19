## Aggregate Tree

Our AggTree is a BallTree variant optimized for high query speeds & reduced memory usage at the sacrifice of accuracy. This tree works best when you have dense regions of points smaller than your bandwidth and large sample sizes. In cases like these, standard ball trees still have to evaluate all points in these dense regions while our AggTree only evaluates a single aggregate node.

You can tune how aggresively nodes are aggregated with the `atol` parameter. When using our AggTree it may be worth comparing how this parameter effects error for your usecase against our ball tree. Our error bounds calcuation can be overly conservitive and the true absolute error for a given usecase will often be lower than this absolute tolerance parameter.

### Benchmarks 

The following heatmap was generated using a bandwidth of 0.05 and an atol of 0.001

<img width="2708" height="855" alt="kde_heatmap_compare" src="https://github.com/Flynn500/ironforest/blob/003842528571875f611550195d533f31fb73f52d/docs/kde_heatmap_compare.png" />


The above heatmap is a best-case scenario for our AggTree. The dataset was generated using scikit-learn's make blobs with a STD of 0.04, just below our bandwidth.

<div align="center">

| dim | bw     | agg_time | sk_time | speedup | max_err% | mean_err% |
|-----|--------|----------|---------|---------|----------|-----------|
| 2   | 0.1468 | 0.062    | 3.665   | 59.36x  | 0.281%   | 0.084%    |
| 4   | 0.2254 | 0.326    | 5.417   | 16.62x  | 0.292%   | 0.156%    |
| 8   | 0.3550 | 0.394    | 6.773   | 17.19x  | 0.545%   | 0.388%    |
| 16  | 0.5216 | 0.562    | 15.067  | 26.82x  | 0.596%   | 0.484%    |


</div>

This benchmark was done on a mixture of uniform noise & scikit-learn's make blobs. Our bandwidth was selected using silverman's rule & an atol value of 0.01 was used. 

These highlight the best scenarios to use our aggregate tree. If your dataset doesn't contain these high density regions that become that can be aggregated, KDE calculations devolve into standard ball tree methods. The error drops to 0, but you miss out on the memory compacting and speed increases our AggTree was designed for.

### Implementation

Our AggTree works on the core principle of trying to reduce our dataset into a series of aggregate nodes. Instead of summing the kernel contributions of significant points we sum a mixture from a smaller set of aggregates alongside any raw data that wasn't aggregated.

Tree construction works the exact same as our standard ball tree, but we stop splitting a node when its approximation error is estimated to be below the user-specified absolute tolerance (`atol`). We then calculate the centroid, variance, 3rd & 4th moments of the point-to-centroid distances. We also compute a worst-case error bound for using the Taylor approximation instead of exact evaluation. If this bound falls below `atol`, the node becomes an aggregate leaf and its children are never created.

The error bounds are kernel-dependent. For the Gaussian kernel, we use a 5th-order Taylor remainder:

$$\epsilon \leq \frac{n}{120} \cdot \sup|K^{(5)}| \cdot \frac{R^5}{h^5}$$

For compact-support kernels (Epanechnikov, Uniform, Triangular), the polynomial part of the kernel has exact finite-order derivatives, so the only source of error is points straddling the support boundary. We bound this as:

$$\epsilon \leq n \cdot \frac{R}{h} \cdot K_{\max}$$

Once aggregate nodes are identified, we recurse through the tree and free all data belonging to them — the only values needed to calculate their contribution are the precomputed moments.

For queries, we recurse through the tree pruning nodes that are too far away to make a meaninful contribution. This works the same as a ball tree until we reach an aggregate node. We use a 4th-order Taylor expansion to approximate the aggregate node's contribution:

$$\hat{K} = n \cdot \left( K(r_c) + \frac{1}{2} K''(r_c) \cdot m_2 + \frac{1}{6} K'''(r_c) \cdot m_3 + \frac{1}{24} K''''(r_c) \cdot m_4 \right)$$

Where $r_c$ is the distance from the query point to the node's centroid, and the moments are:

$$m_2 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^2, \quad m_3 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^3, \quad m_4 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^4$$
