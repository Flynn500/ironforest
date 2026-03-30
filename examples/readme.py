import ironforest as irn

dims = 64
n = 100_000
k = 10

#Randomly generate data
gen = irn.random.Generator.from_seed(0)
data = gen.uniform(0.0, 100.0, [n, dims])
query_point = ([50.0] * dims)

#create a spatial index object
index = irn.SpatialIndex(data, tree_type="auto")

#the tree-type, automatically selected by our spatial index based on the dataset.
print(index.tree_type)

result = index.query_knn(query_point, k=k)

#k nearest neighbours
for output_idx, original_idx in enumerate(result.indices):
    print(f"index: {original_idx}, dist: {result.distances[output_idx]}")

#print mean, meadian and max distances
print(f"{result.mean():.2f}, {result.median():.2f}, {result.radius():.2f}")


