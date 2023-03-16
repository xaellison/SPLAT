using Clustering, Statistics

"""
Shitty bounding volume method.
1. Apply k-means on circumcenters of Triangles  This gives you cluster locations, 
    and triangle memberships.
2. For each cluster, find the smallest sphere enclosing the cluster's members'
    circumscribing spheres
3. Put each Triangle into each cluster where the circumscribing sphere of the
    triangle overlaps with the bounding sphere of the cluster.
"""
function cluster_fuck(tris, N)
   # @info "Clustering $(length(tris)) tris"
    # annoying matrix for clustering
    data = zeros(Float32, (3, length(tris)))
    for (n, T) in enumerate(tris)
        data[:, n] = circumcenter(T)
    end
    
    result = kmeans(data, N, tol=1e5)
    
    # initialize clusters so each Tri in one cluster
    clusters = Dict()
    for (n, T) in enumerate(tris)
        c = result.assignments[n]
        if ! (c in keys(clusters))
            clusters[c] = Set()
        end
        push!(clusters[c], T)
    end
    ℜ³_centers = ℜ³.(result.centers[1, :], result.centers[2, :], result.centers[3, :])
    #@info Set(result.assignments)
    radii = [maximum(norm(circumcenter(T) - ℜ³_centers[c]) + circumscribing_sphere(T).radius for T in clusters[c]) for c in 1:N]
    spheres = Sphere.(ℜ³_centers, radii)
    sort!(spheres)
    # the clusters need to be updated to include Triangles which ma
    for T in tris
        for (c, S) in enumerate(spheres)
            # it shouldn't be possible to have an empty cluster at this point but ya never know
            if ! (c in keys(clusters))
                clusters[c] = Set()
            end
            S_T = circumscribing_sphere(T)
            if norm(S_T.origin - S.origin) <= S_T.radius + S.radius
                push!(clusters[c], T)
            end
        end
    end
    @assert foldl(union, values(clusters)) == Set(tris)
    @info "overcount factor = $(sum(map(length, values(clusters))) / length(tris))"

    inverted_indices = Dict(T => n for (n, T) in enumerate(tris))
    clusters_by_index = Dict()

    for c in keys(clusters)
        # sort should help the gpu have better aligned accesses in certain instances (eg, two well separated models)
        clusters_by_index[c] = sort([inverted_indices[T] for T in clusters[c]])
    end

    return spheres, clusters_by_index
end

function load_and_cluster_fuck(path::String, N)
    tris = mesh_to_FTri(load(path))
    cluster_fuck(tris, N)
end

function spatial_partition(tris, depth::Int=0)
    # returns a list of views of `tris` bisected by half-planes `depth` times, with the planes' orientations rotating with `depth`
    if depth < 0
        return error("negative recursion")
    end
    
    if depth == 0
        return [tris]
    end

    dimension = depth % 3 + 1
    mid_value = median(mean(T[n][dimension] for n in 2:4) for T in tris)

    branch_indices = map(T -> minimum(T[n][dimension] for n in 2:4) <= mid_value, tris)
    tri_view = @view tris[branch_indices]
    lefties = spatial_partition(tri_view, depth - 1)

    branch_indices = map(T -> maximum(T[n][dimension] for n in 2:4) >= mid_value, tris)
    tri_view = @view tris[branch_indices]
    righties = spatial_partition(tri_view, depth - 1)
    
    return vcat(lefties, righties)
end

function bv_partition(tris, depth=6; verbose::Bool=false)
    partitions = spatial_partition(Array(tris), depth)
	bounding_volumes = collect(BVBox(ℜ³(
								minimum(minimum(p[1] for p in t[2:4]) for t in tris), 
								minimum(minimum(p[2] for p in t[2:4]) for t in tris), 
								minimum(minimum(p[3] for p in t[2:4]) for t in tris), 
							),
							ℜ³(
								maximum(maximum(p[1] for p in t[2:4]) for t in tris), 
								maximum(maximum(p[2] for p in t[2:4]) for t in tris), 
								maximum(maximum(p[3] for p in t[2:4]) for t in tris), 
							)) for tris in partitions)
	bounding_volumes_members = Dict(n => p.indices[1] for (n, p)  in enumerate(partitions))

    if verbose
        @info """
                Bounding Volume Summary:
                Tri count = $(length(tris))
                Number of partitions = $(length(partitions))
                Overcount factor = $(sum(map(length, partitions)) / length(tris))
                Mean partition size = $(mean(map(length, partitions)))    
                Median partition size = $(median(map(length, partitions)))    
                Min partition size = $(minimum(map(length, partitions)))    
                Max partition size = $(maximum(map(length, partitions)))    
                Std partition size = $(std(map(length, partitions)))    
            """
    end

    return bounding_volumes, bounding_volumes_members
end

#@time load_and_cluster_fuck("objs\\artemis_smaller.obj", 3);

