include("../src/geo.jl")
# Unit tests!

@test vector_cosine(V3(1, 0, 0), V3(0, 1, 0)) == 0
@test vector_cosine(V3(1, 1, 1), V3(0, 1, 0)) ≈ 1 / sqrt(3)
@test vector_cosine(V3(1, 1, 1), V3(0, 2, 0)) ≈ 1 / sqrt(3)

@test project(V3(1, 0, 0), V3(0, 1, 0)) ≈ V3(0, 0, 0)
@test project(V3(1, 0, 0), V3(1, 2, 0)) ≈ V3(1, 2, 0) * (1 / sqrt(5))

@test reflect(V3(-1, -1, 0), V3(0, 1, 0)) ≈ V3(-1, 1, 0)
@test reflect(V3(-1, -1, 0), V3(0, -1, 0)) ≈ V3(-1, 1, 0)
@test reflect(V3(-1, -1, 0), V3(1, 0, 0)) ≈ V3(1, -1, 0)
@test reflect(V3(-1, -1, 0), V3(-1, 0, 0)) ≈ V3(1, -1, 0)
@test reflect(V3(-4, -1, 0), V3(0, 1, 0)) ≈ V3(-4, 1, 0)
@test reflect(V3(-1, -4, 0), V3(0, 1, 0)) ≈ V3(-1, 4, 0)

# more complex test case that caused an actual bug
d0 = V3(-0.0993602, -0.990078, 0.0993602)
n = V3(0.21059, 0.310929, -0.926809)
d1 = V3(0.077896, -0.728365, -0.680746)
@test norm(reflect(d0, n) - d1) < 1e-6

# test normal's sign doesn't matter
@test distance_to_plane(V3(1, 2, 4), V3(2, 3, 1), V3(-2, 0, -2), V3(8, 1, 0)) ==
      distance_to_plane(V3(1, 2, 4), V3(2, 3, 1), V3(-2, 0, -2), V3(8, 1, 0) * -1)
# test that negative direction gives negative distance
@test distance_to_plane(V3(1, 2, 4), V3(2, 3, 1) * -1, V3(-2, 0, -2), V3(8, 1, 0)) ==
      distance_to_plane(V3(1, 2, 4), V3(2, 3, 1), V3(-2, 0, -2), V3(8, 1, 0)) * -1
