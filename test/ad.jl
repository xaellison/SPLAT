# diamond around origin
a = V3(1, 0, 0)
b = V3(0, 1,0)
c = -a
d = -b
na=a
nc=c
nb=nd=V3(0,0,1)
t1 = STri(normalize(cross(a-b,b-d)), a, b, d, na, nb, nd)
t2 = STri(normalize(cross(c-b,b-d)), c, b, d, nc, nb, nd)
θ = pi / 6
ϕ = pi / 3
P(θ, ϕ) = V3(1,0,0)*sin(θ)*cos(ϕ)+V3(0,1,0)*sin(θ)*sin(ϕ) - V3(0,0,1)*cos(θ)
D(θ, ϕ) = -P(θ, ϕ)

# ray from point on unit sphere to origin. Dir varies but pos doesn't
R = ADRay(P(θ, ϕ),
           zero(V3),
           zero(V3),
           zero(V3),
           D(θ, ϕ),
           zero(V3),
           ForwardDiff.derivative(θ->D(θ, ϕ), θ),
           ForwardDiff.derivative(ϕ->D(θ, ϕ), ϕ),
           true,
           -1,
           1,
           0.0f0,
           zero(V3),
           false)


dx = ForwardDiff.derivative(x->p(R, t1, 0.0f0, x, 0.0f0), 0.0f0)
dy = ForwardDiff.derivative(y->p(R, t1, 0.0f0, 0.0f0, y), 0.0f0)
@assert dot(cross(dx, dy), V3(0, 0, 1)) ≈ norm(cross(dx, dy))

# ray from point on unit sphere to origin. Dir and pos vary
R = ADRay(P(θ, ϕ),
           zero(V3),
           ForwardDiff.derivative(θ->P(θ, ϕ), θ),
           ForwardDiff.derivative(ϕ->P(θ, ϕ), ϕ),
           D(θ, ϕ),
           zero(V3),
           ForwardDiff.derivative(θ->D(θ, ϕ), θ),
           ForwardDiff.derivative(ϕ->D(θ, ϕ), ϕ),
           true,
           -1,
           1,
           0.0f0,
           zero(V3),
           false)

 dx = ForwardDiff.derivative(x->p(R, t1, 0.0f0, x, 0.0f0), 0.0f0)
 dy = ForwardDiff.derivative(y->p(R, t1, 0.0f0, 0.0f0, y), 0.0f0)
 @assert dx ≈ zero(V3) && dy ≈ zero(V3)
