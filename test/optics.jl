adr = ADRay(V3(-1, 0, 0),
            zero(V3),
            V3(1, 0, 0),
            zero(V3),
            true,
            -1,
            -1,
            550.0f0,
            RAY_STATUS_ACTIVE)

a = V3(1, 1, 1)
b = V3(0, -2, 1)
c = V3(0, 1, -2)
n = normalize(cross(a-b, b-c))

t = FTri(n, a, b, c, n, n, n, zero(V3), zero(V3), zero(V3))

adr2 = evolve_ray(adr, Int32(1), t, 1.0f0, 2)
