function warp_min_sync(mask::UInt32, val::UInt32)
llvm_code = """declare i32 @llvm.nvvm.redux.sync.umin(i32, i32)
    define i32 @redux_sync_min_u32(i32 %src, i32 %mask) {
    %val = call i32 @llvm.nvvm.redux.sync.umin(i32 %src, i32 %mask)
    ret i32 %val
    }
    """
    return Base.llvmcall((llvm_code, "redux_sync_min_u32"),
                            UInt32,
                            Tuple{UInt32, UInt32},
                            val,
                            mask)
end
"""
declare i32 @llvm.nvvm.redux.sync.umax(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_max_u32
define i32 @redux_sync_max_u32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.max.u32
  %val = call i32 @llvm.nvvm.redux.sync.umax(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.add(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_add_s32
define i32 @redux_sync_add_s32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.add.s32
  %val = call i32 @llvm.nvvm.redux.sync.add(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.min(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_min_s32
define i32 @redux_sync_min_s32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.min.s32
  %val = call i32 @llvm.nvvm.redux.sync.min(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.max(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_max_s32
define i32 @redux_sync_max_s32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.max.s32
  %val = call i32 @llvm.nvvm.redux.sync.max(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.and(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_and_b32
define i32 @redux_sync_and_b32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.and.b32
  %val = call i32 @llvm.nvvm.redux.sync.and(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.xor(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_xor_b32
define i32 @redux_sync_xor_b32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.xor.b32
  %val = call i32 @llvm.nvvm.redux.sync.xor(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.or(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_or_b32
define i32 @redux_sync_or_b32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.or.b32
  %val = call i32 @llvm.nvvm.redux.sync.or(i32 %src, i32 %mask)
  ret i32 %val
}
"""



function reduce_k(c)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  v = c[i]
  v2 = CUDA.shfl_down_sync(0xFFFFFFFF, v, 1)
  v = min(v, v2)
  v2 = CUDA.shfl_down_sync(0xFFFFFFFF, v, 2)
  v = min(v, v2)
  v2 = CUDA.shfl_down_sync(0xFFFFFFFF, v, 4)
  v = min(v, v2)
  v2 = CUDA.shfl_down_sync(0xFFFFFFFF, v, 8)
  v = min(v, v2)
  v2 = CUDA.shfl_down_sync(0xFFFFFFFF, v, 16)
  v = min(v, v2)
  v = CUDA.shfl_sync(0xFFFFFFFF, v, 1)
  c[i] = v
  return
end
  

function redux_k(c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    v = c[i]
    v = warp_min_sync(0xffffffff, v)
    c[threadIdx().x] = v
    return
end

c = CUDA.rand(UInt32, 1<<20)
c_old = c |> copy |> Array
@btime CUDA.@sync @cuda threads = 1024 blocks=length(c)÷1024 redux_k(c)
#minimum(c_old) == c[1]