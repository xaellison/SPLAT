
function sky_color(dir, λ, phi) ::Float32
	d = normalize(dir)
	out = 0.1+(0.7*(d[3] ^ 3 * 0.5 + 0.5))
	if isnan(out)
		return 0.0f0
	end
	return out
end


function sky_color_rezz(dir :: AbstractArray{T}, λ::T, phi::T)::T where T
	t1 = atan(dir[1], sqrt(dir[2]^2+dir[3]^2))
	t2 = atan(dir[2], sqrt(dir[1]^2+dir[3]^2))
	t3 = atan(dir[3], sqrt(dir[1]^2+dir[2]^2))
	if t1 < 0
		t1 -= phi
	else
		t1 += phi
	end
	if t2 < 0
		t2 -= phi
	else
		t2 += phi
	end
	if t3 < 0
		t3 -= phi
	else
		t3 += phi
	end
	if t1 < 0
		t1 += pi
	end
	if t2 < 0
		t2 += pi
	end
	if t3 < 0
		t3 += pi
	end
	k = 2*pi/80
	if xor(abs(t1) % k < k/2, abs(t2) % k < k/2, abs(t3) % k < k/2)
		return 1.0
	end
	return 0.0
end



function sky_rings(dir :: AbstractArray{T}, λ::T, phi::T)::T where T
	t1 = atan(dir[1], sqrt(dir[2]^2+dir[3]^2))

	if t1 < 0
		t1 -= phi
	else
		t1 += phi
	end

	k = 2 * pi / 20
	if abs(t1) % k < k/2
		return 1.0
	end
	return 0.0
end

function sky_curtains(dir :: AbstractArray{T}, λ::T, phi::T)::T where T
	t1 = atan(dir[2], sqrt(dir[3]^2+dir[1]^2))

	if t1 < 0
		t1 -= phi
	else
		t1 += phi
	end

	k = 2 * pi / 20
	if abs(t1) % k < k/2
		return 1.0
	end
	return 0.0
end


function sky_stripes_down(dir :: AbstractArray{T}, λ::T, phi::T)::T where T
	t1 = atan(dir[3], sqrt(dir[1]^2+dir[2]^2))

	t1 += phi

	k = 2 * pi / 20
	if (t1 + pi) % k < k/2
		return 1.0
	end
	return 0.0
end

function solid_angle_intensity(r :: ADRay) :: Float32
	"""
	Find the real hit pos, get the interpolated normal
	treat as new triangle with actual normal the interpolated
	"""


	T = STri(r.last_normal,
			 r.pos,
			 r.pos + cross(r.last_normal, V3(1,0,0)),
			 r.pos + cross(r.last_normal, cross(r.last_normal, V3(1,0,0))),
			 r.last_normal, r.last_normal, r.last_normal)
	dx = ForwardDiff.derivative(x->p(r, T, r.λ, x, 0.0f0), 0.0f0)
	dy = ForwardDiff.derivative(y->p(r, T, r.λ, 0.0f0, y), 0.0f0)

	norm(cross(dx, dy))

end

function shade(r :: ADRay, sky :: S, λ, ϕ) :: Float32 where S
	r.in_medium ? 0.0f0 : 0.5f0 * solid_angle_intensity(r) * 1e-2
end

"""
I don't think it's possible to get continuous lens like shading  on a finite mesh.
The brightness is based on a derivative of direction. For shading to be "smooth",
(but really continuous in terms of a curve) this derivative must be continuous.

An arc of ray's direction is continuous but not smooth due to a finite mesh.
The derivative of that direction is then not guaranteed to be continuous, hence the
color suddenly "jumps" at mesh edges.

This is also a good time to note I've been wrong by estimating radiance inbound
as a function of the derivative of direction. Rather, it should be of position,
the effective area absorbing even if only from one direction (etendre). Direction
may not suffer the differentiability problem, but position most certainly does.

Find the real hit pos, get the interpolated normal
treat as new triangle with actual normal the interpolated

I thought this would be clever but not enough
"""
