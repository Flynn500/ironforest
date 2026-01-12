# Substratum
Substratum is a rust package designed for the dubious python library. Dubious is a project with the personal constraint that no libraries can be used other than numpy. Substratum is my attempt to rewrite some of numpy's features in hopes that it can eventually replace numpy as dubious's lone dependency. I also intend to expand Substratum to include more features, although it is not intended to be a full fledged numpy replacement.

## NdArray
NdArray / Array in our python bindings as an N-dimensional array object. This array supports most broadcast operations alongside a hanful of matrix creation constructors. Matrix operations will be supported soon.

## Generator
Substratum provides a random generator object that can sample from uniform, normal, gamma and beta distributions. Support for additional distributions is planned.

## Examples
```python
import substratum as ss

a = ss.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
b = ss.Array([2, 2], [5.0, 6.0, 7.0, 8.0])

print(f"a + b = {(a + b).tolist()}")
print(f"a * b = {(a * b).tolist()}")
```
Output: 
```python
import substratum as ss

gen = ss.Generator.from_seed(123)

uniform = gen.uniform(0.0, 1.0, [2, 3])
print(f"Uniform [0, 1): {uniform.tolist()}")

normal = gen.standard_normal([2, 3])
print(f"Standard normal: {normal.tolist()}")
```
Output:
