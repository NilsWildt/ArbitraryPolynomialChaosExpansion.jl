using Test
using BenchmarkTools
using TimerOutputs
using Distances


function pairdist1(a, b; doroot=true)
        na = size(a, 1)
        nb = size(b, 1)
        r = a * b'
        sa2 = sum(a .^ 2, dims=2)
        sb2 = sum(b .^ 2, dims=2)
        @inbounds for j = 1:nb
                @simd for i = 1:na
                        r[i, j] = @views sa2[i] + sb2[j] - 2 * r[i, j]
                        if doroot
                                r[i, j] = @views isnan(r[i, j]) ? NaN : sqrt(max(r[i, j], 0.0))
                        end
                end
        end
        return r
end


function pairdist(a, b; doroot=true)
        na = size(a, 1)
        nb = size(b, 1)
        r = a * b'
        sa2 = sum(a .^ 2, dims=2)
        sb2 = sum(b .^ 2, dims=2)
        @inbounds for j = 1:nb
                @simd for i = j:na
                        r[i, j] = @views sa2[i] + sb2[j] - 2 * r[i, j]
                        if doroot
                                r[i, j] = @views isnan(r[i, j]) ? NaN : sqrt(max(r[i, j], 0.0))
                        end
                end
                @simd for i = 1:j-1
                        r[i, j] = r[j, i]
                end
        end


        return r
end

function pairdist_threaded(a, b; doroot=true)
        na = size(a, 1)
        nb = size(b, 1)
        r = a * b'
        sa2 = sum(a .^ 2, dims=2)
        sb2 = sum(b .^ 2, dims=2)
        Threads.@threads for j = 1:nb
                @simd for i = j:na
                        r[i, j] = @views sa2[i] + sb2[j] - 2 * r[i, j]
                        if doroot
                                r[i, j] = @views isnan(r[i, j]) ? NaN : sqrt(max(r[i, j], 0.0))
                        end
                end
                @simd for i = 1:j-1
                        r[i, j] = r[j, i]
                end
        end


        return r
end


# Something wrong?
# function pairdist_slow(x, y; doroot=true)
#         n = size(x, 1)
#         XY = kernel_dot(x, x)
#         X² = sum(x .^ 2; dims=2)
#         X²rep = reshape(repeat(X², (n)), (n, n))
#         D = (X²rep + X²rep' - 2 * XY)
#         if doroot
#                 D = sqrt.(max.(D, 0.0))
#         end
#         return D
# end

@fastmath function kernel_dot(X::AbstractArray, Y::AbstractArray)
        dimY = size(Y)
        dimX = size(X)
        Lx = Int64(dimX[1])
        Ly = Int64(dimY[1])
        A = zeros(Float64, (Lx, Ly))
        @inbounds @simd for i in 1:Ly
                for j in 1:Lx
                        A[i, j] = @views X[i] * Y[j]
                end
        end
        # display(A)
        return A
end




function pairdist_tullio(x::AbstractArray{<:AbstractFloat,3}, y::AbstractArray{<:AbstractFloat,3})
        @show "test"
        @tullio z[i, j] := sqrt((x[1, i] - y[1, j])^2 + (x[2, i] - y[2, j])^2 + (x[3, i] - y[3, j])^2)
        return z
end

function pairdist2_tullio(x::AbstractArray{<:AbstractFloat,3}, y::AbstractArray{<:AbstractFloat,3})
        @tullio z[i, j] := (x[1, i] - y[1, j])^2 + (x[2, i] - y[2, j])^2 + (x[3, i] - y[3, j])^2
        return z
end

function pairdist_3D(x::AbstractArray, y::AbstractArray)
        @tullio z[i, j] := sqrt((x[1, i] - y[1, j])^2 + (x[2, i] - y[2, j])^2 + (x[3, i] - y[3, j])^2)
        return z
end


const to = TimerOutput()
reset_timer!(to)
for i in 1:1000
        @show i
        x = rand(500, 3)
        y = copy(x)
        @timeit to "pdist 1" b = pairdist1(x, y)
        @timeit to "pdist slow " c = pairdist_slow(x, y)
        @timeit to "pdist thread" d = pairdist_threaded(x, y)
        # @timeit to "pdist normal" e = pairdist(x, y)
        @timeit to "pdist tullio" f = pairdist_tullio(x', y')


end
to
x = rand(3,5)
y = rand(3,5)

a = Distances.pairwise(Euclidean(), x, y)
f = pairdist_t(x, y)

x = x'
y = y'

b = pairdist1(x, y)
# c = pairdist_slow(x, y)
d = pairdist_threaded(x, y)




a = Distances.pairwise(Euclidean(), x, y)

# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  35.000 μs …  25.085 ms  ┊ GC (min … max):  0.00% … 99.39%
#  Time  (median):     63.800 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   83.671 μs ± 393.506 μs  ┊ GC (mean ± σ):  11.49% ±  2.62%

#    █▂▂▅▃▂▄▂     
#   ▅████████▇▇█▇██▇▇▅▄▄▃▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂ ▃
#   35 μs           Histogram: frequency by time          252 μs <

#  Memory estimate: 80.05 KiB, allocs estimate: 5.

f = pairdist_tullio(x, y)

# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  11.500 μs …  39.309 ms  ┊ GC (min … max):  0.00% … 0.00%
#  Time  (median):     24.800 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   49.842 μs ± 602.748 μs  ┊ GC (mean ± σ):  22.37% ± 2.63%

#      ▇█▅▁                  
#   ▂▄▇████▅▅▅▆▅▆▄▃▂▂▂▃▄▅▆▆▆▆▄▄▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂ ▃
#   11.5 μs         Histogram: frequency by time          105 μs <

#  Memory estimate: 78.31 KiB, allocs estimate: 10.