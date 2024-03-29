function glass(λ ::T) :: T where T

	λ1 = 400.0f0
	λ2 = 700.0f0
	n1 = 1.3f0
	n2 = 1.4f0
	m = (n2 - n1) / (λ2 - λ1)
	return m * (λ - λ1) + n1
end

function air(λ::T) :: T where T
	return one(T)
end
