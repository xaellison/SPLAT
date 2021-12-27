glass(λ :: Float32) :: Float32 = begin
							       λ1 = 400.0f0
								   λ2 = 700.0f0
								   n1 = 1.5f0
								   n2 = 1.7f0
								   m = (n2 - n1) / (λ2 - λ1)
								   return m * (λ - λ1) + n1
								end
