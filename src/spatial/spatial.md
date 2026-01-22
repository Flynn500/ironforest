# Spatial Module
The spatial module holds a variety of tree structures, each possesing kNN, raidus and KDE queries. These queries function nearly the same in each, although pruning rules differ between. 

## Optmizations
The biggest speedup I've implemented so far was making the trees more cache friendly. Previously the data array remained untouched, while we manipulated an index array to deal with in-tree computations. This seemed fine in principle as we want to return the indices as our result, but it is not cache friendly. After adding a reorder function we increased speeds by 30% across queries. This function just rearranges our data vector so that nodes close to each other a stored nearby. This makes it easier for the CPU to cache values as we aren't jumping to random points in our arrays.

KDE Approximation: This is somewhat home-baked, I'm sure it has been done before but I am unsure on its statistical correctness. Our KDE approximation works by two criteria, min_samples or span. If a node has either a span or child count less than these values, we use a single kernel for the entire node (taking into account its span and number of children). This works very well for the particular use case I needed it for, where we had regions of very dense points that could easily be treated as a single kernel instead of summing to contributions of each point individually. 
