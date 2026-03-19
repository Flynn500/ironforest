import ironforest as irn

dims = 256
n = 100_000
k = 10

#Randomly generate data
gen = irn.random.Generator.from_seed(0)
data = gen.uniform(0.0, 100.0, [n, dims])
query_point = ([50.0] * dims)

#Use random projections to reduce the dimensionality
reducer, points = irn.spatial.ProjectionReducer.fit_transform(data, 50)
tree = irn.spatial.KDTree.from_array(points, leaf_size=50)

#Use our reducer to convert the query
query_point = reducer.transform(query_point)
result = tree.query_knn(query_point, k=k)

#k nearest neighbours
for output_idx, original_idx in enumerate(result.indices):
    print(f"index: {original_idx}, dist: {result.distances[output_idx]}")

#print mean, meadian and max distances
print(f"{result.mean():.2f}, {result.median():.2f}, {result.radius():.2f}")
