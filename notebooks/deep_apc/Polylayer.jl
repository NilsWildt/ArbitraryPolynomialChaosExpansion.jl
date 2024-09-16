"""
Define a custom Polynomial layer in Lux.jl to allow the polynomial coefficients to 
be estimated through training. 
"""

struct Polylayer{F} <: Lux.AbstractExplicitLayer
    out_dims::Int
    degree::Int
    init_weight::F
end

function Base.show(io::IO, d::Polylayer)
    println(io, "Polylayer(out_dims: $(d.out_dims)), degree: $(d.degree)")
end

function Polylayer(out_dims::Int, degree::Int, init_weight=Lux.rand32)
    dtypes = (typeof(init_weight))
    return Polylayer{dtypes}(out_dims, degree, init_weight)
end

# Allow arbitrary order for arguments with default values
function Polylayer(; out_dims::Int=2, degree::Int=2, init_weight=Lux.rand32)
    dtypes = (typeof(init_weight))
    return Polylayer{dtypes}(out_dims, degree, init_weight)
end

"""
Evaluate the polynomial whose coefficients are given by ps.coeffs (ps.coeffs[1] is lowest degree coefficient)
ps.coeffs is a matrix of size [:, degree]

What to do when there are batches of points? 

`ps` might not be used
If I write T <: AbstractArray{Float64}, the function will not work with Float32. So I choose to 
maintain flexibility. Put AbstractArray{Float64} would be useful if I am interested in working fully in 
Float64. On the GPU, I would write AbstractArray{Float32} for efficiency reasons on the GPU.
"""
function (l::Polylayer)(x::T, ps, st) where {T<:AbstractArray}
    if (l.out_dims == 1)
        poly = generate_polynomial_1D(l.degree)
        return poly(ps.coeffs, x), st
    elseif (l.out_dims == 2)
        x1 = x[1, :]
        x2 = x[2, :]
        N = size(x, 2)

        # Should ideally be generated only once and not every time I call the model
        # Coef generated within generated_polynomial. Should not be

        poly1 = generate_polynomial_2D(l.degree)
        poly2 = generate_polynomial_2D(l.degree)

        # poly1(coef, x, y) returns an arra of size (1,N), for N points

        # Use views to reduce copying? Not worth is for small arrays
        # Must make more efficient!
        # ????? CORRECT?
        coef1 = view(ps.coeffs, :, 1)  # ps.coeffs is (nb_terms, out_dims)
        coef2 = view(ps.coeffs, :, 2)
        p1 = poly1(coef1, x1, x2)  # should already be of size (1, N)
        p2 = poly2(coef2, x1, x2)
        # println("poly layer 2D: 1. sizes p1, p2: ", size(p1), ", ", size(p2)) # should be (1, N)
        # println("poly layer 2D: before reshape, p1 size: ", size(p1))
        p1 = reshape(p1, 1, N)  # should not be required (one row)
        p2 = reshape(p2, 1, N)
        # println("polylayer 2D: after reshape, p1 size: ", size(p1))
        p3 = vcat(p1, p2)   # p3: (2, nb_coefs)
        # println("polylayer, size p3: ", size(p3))  # (2, 1)
        return vcat(p1, p2), st  # size: (2,nb_coefs) (no need to return st)
    else
        println("Poly_layer: l.out_dims > 2 not implemented")
    end
end

# ! The next four functions have the Lux prefix to make sure dynamic dispaching works properly
# overload initialparameters in the Lux environment
# coeffs[out_dims, degree+1]

# Probably unique in my project. 
# What happens if I do not prefix by Lux and I use the polylayer inside some other module. It is not clear this will work.
function nb_coefs_per_poly(d::Polylayer)
    if d.out_dims == 1
        return d.degree + 1
    elseif d.out_dims == 2
        return Int((d.degree + 1) * (d.degree + 2) / 2.0)  # (very specific). Works because there is no loss of precision
    else
        println("out_dims greater 2 not implemented.")
    end
end

function Lux.initialparameters(rng::AbstractRNG, d::Polylayer)
    nb = nb_coefs_per_poly(d)
    # why the final comma?
    if (d.out_dims == 1)
        return (coeffs=d.init_weight(rng, nb),)
    else
        return (coeffs=d.init_weight(rng, nb, d.out_dims),)
    end
end

# overload initialstates in the Lux environment
Lux.initialstates(::AbstractRNG, ::Polylayer) = NamedTuple()

Lux.statelength(d::Polylayer) = 0

function Lux.parameterlength(d::Polylayer)
    return nb_coefs_per_poly(d) * d.out_dims
end
# ================== END DEFINITION of Polynomial Layer =======================================
# Simple standalone code to test the polynomial layer. Must be part of this file. 
function test_polynomial_layer(dim)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    # rand32(rng, x) = 0.5 * (1. - 2 * Lux.rand32(rng, x))

    if (dim == 1)
        # 1D layer
        out_dims = 1
        degree = 3
        # The weights are initialized to zero
        model = Polylayer(; out_dims=out_dims, degree=degree, init_weight=Lux.zeros32)

        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps)
        Lux.initialparameters(rng, model)

        Lux.ComponentArray(ps)
        x = rand(128)  # data

        # Evaluate the layer
        y, _ = model(x, ps, st)  # returns a single scalar
        return y

    elseif (dim == 2)
        # 2D layer
        out_dims = 2
        degree = 3
        model = Polylayer(; out_dims=out_dims, degree=degree, init_weight=Lux.zeros32)

        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps)
        Lux.initialparameters(rng, model)

        Lux.ComponentArray(ps)
        x = rand(2, 128)  # data

        # Evaluate the layer
        println("test_polynomial_layer 2D, ps: ", ps.coeffs)
        y, _ = model(x, ps, st)
        return y
    end
end