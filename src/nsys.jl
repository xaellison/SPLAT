using CUDA

 let 

    c = CUDA.rand(1000)
    d = CUDA.rand(1000)
    sum(c + d)
    
end