using BenchmarkTools
using AirspeedVelocity
using PkgJogger
using Random
import APCE.aPCE_OrthonormalBasis
import APCE.aPCE_MultivariatePolynomialDegrees
import APCE.partitionTrainTest

using DrWatson
using MAT
const SUITE = BenchmarkGroup()

SUITE["main"] = BenchmarkGroup()


function benchmark_basis(file)
     FT = Float64
     #  Loading Input distributions, Training Data and Validation Data 
		print(keys(file))
		# Extracting the variables
		# Input_distributions = file["Xtr"] |> Array{FT}
		indall = 1:1000
		rng = Xoshiro(42)
		itrain,itest = partitionTrainTest(indall;at= 0.75, rng )
		TrainingInput = file["Xtr"][itrain, :] |> Array{FT}
		TrainingOutput = file["Ytr"][itrain, 60:62] |> Array{FT}
		@info "" size(TrainingInput)

		ValidationInput = file["Xtr"][itest, :] |> Array{FT}
		ValidationOutput = file["Ytr"][itest, 60:62] |> Array{FT}
        degree = 7
        Φ = aPCE_OrthonormalBasis(TrainingInput, degree)
end

SUITE["main"]["benchmark_orthonomrmal_basis"] = @benchmarkable benchmark_basis(file) setup = (file = matread(datadir("training_data.mat"))) evals = 1 samples =  10

function benchmark_MultivariatePolynomialDegrees(file)
    aPCE_MultivariatePolynomialDegrees(num_dimensions,max_degree; qnorm = 1.0)
end

		
SUITE["main"]["MultivariateDegrees"] = @benchmarkable aPCE_MultivariatePolynomialDegrees(5,5) setup = () evals = 1 samples =  10

run(SUITE)

begin 
    import AirspeedVelocity._get_script
    function _get_script(;
        package_name::String,
        benchmark_on::Union{Nothing,String}=nothing,
        url::Union{Nothing,String}=nothing,
        path::Union{Nothing,String}=nothing,
    )::Tuple{String,Union{String,Nothing}}
        # Create temp env, add package, and get path to benchmark script.
        @info "Downloading package's latest benchmark script, assuming it is in benchmark/benchmarks.jl"
        if benchmark_on !== nothing
            @info "Downloading from $benchmark_on."
        end
        tmp_env = mktempdir(; cleanup=false)
        to_exec = quote
            ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
            using Pkg
            Pkg.add(
                PackageSpec(; name=$package_name, rev=$benchmark_on, url=$url, path=$path);
                io=devnull,
            )
            using $(Symbol(package_name)): $(Symbol(package_name))
            root_dir = dirname(dirname(pathof($(Symbol(package_name)))))
            open(joinpath($tmp_env, "package_path.txt"), "w") do io
                write(io, root_dir)
            end
        end
        path_getter = joinpath(tmp_env, "path_getter.jl")
        open(path_getter, "w") do io
            println(io, to_exec)
        end
        run(`C:\Users\wildt\.julia\juliaup\julia-1.10.2+0.x64.w64.mingw32\bin\julia.exe --project="$tmp_env" --startup-file=no "$path_getter"`)

        root_dir = readchomp(joinpath(tmp_env, "package_path.txt"))
        script = joinpath(root_dir, "benchmark", "benchmarks.jl")
        if !isfile(script)
            @error "Could not find benchmark script at $script. Please specify the `script` manually."
        end
        @info "Found benchmark script at $script."
        maybe_project_toml = joinpath(root_dir, "benchmark", "Project.toml")
        project_toml = if isfile(maybe_project_toml)
            @info "Found Project.toml at $maybe_project_toml."
            maybe_project_toml
        else
            nothing
        end

        return script, project_toml
    end

end
benchmark("APCE","dirty";path=raw"\\iws-ls3-cifs.tik.uni-stuttgart.de\shared\users\ac125867\03_projects\32_aPC_julia\APCE.jl",output_dir="benchmark")