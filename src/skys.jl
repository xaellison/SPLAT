
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

function solid_angle_intensity(r :: ADRay)
	if norm(r.pos_x′) == 0 || norm(r.pos_y′) ==0
#		return 1.0f0
	end
	return norm(cross(r.dir_x′, r.dir_y′)) ^ 0.25 #* norm(cross(r.pos_x′, r.pos_y′))
	#cbrt(abs(dot(cross(r.dir_x′, r.dir_y′),  cross(r.pos_x′, r.pos_y′))))
end

function shade(r :: ADRay, sky :: S, λ, ϕ) :: Float32 where S
	r.in_medium ? 0.0f0 : sky(ray_dir(r, λ, 0.0f0, 0.0f0), λ, ϕ) * solid_angle_intensity(r)
end
