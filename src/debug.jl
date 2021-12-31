r=ADRay(V3(1,1,1), V3(1,1,1), V3(1,1,1), V3(1,1,1), true, -1, 1, 550.0f0)
dnt = ((1.0f0, 1.0f0), Int(1), Tri(V3(1,1,1), V3(1,1,1), V3(1,1,1),  V3(1,1,1)))



function debug2(v, N, λ)
  d= refract(v, N, air(λ), glass(λ))
  n1, n2 = air, glass
  if λ > 0.5
    n1, n2 = n2, n1
  end
    dprime = ForwardDiff.derivative(λ -> refract(v, N, n1(λ), n2(λ)), λ)
  return d, dprime
end
