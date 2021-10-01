using MLJ
using MLJBase
using DataFrames
using Random
using CSV
using CategoricalArrays
using DrWatson

include("src/pdMISVMClassifier.jl")

Random.seed!(42)

function load_musk()
    # Load the musk dataset from UCI
    musk_df = CSV.read(datadir("musk2.data"), DataFrame, header=false)
# Convert integer columns to float for normalization
    cols = 3:168
    cols2float = names(musk_df)[cols]
    [musk_df[!, col] = musk_df[!, col] * 1.0 for col in cols2float]

    # Standarize input data from musk
    X_standardizer = Standardizer()
    X = MLJBase.transform(fit!(machine(X_standardizer, musk_df)), musk_df)

    # Group data by molecule (e.g. Column1) and create bags
    gdf = groupby(X, :Column1)
    Xs = [DataFrame(m[!, cols]) for m in gdf]
    ys = CategoricalArray([startswith(m.Column1[1], "MUSK") for m in gdf])
    ys = recode(ys, true=>"MUSK", false=>"NON-MUSK")
    ordered!(ys, true)

    return Xs, ys
end

println("Loading MUSK-2 Dataset...")
X, y = load_musk()
println("Finished Loading MUSK-2.")

print("Verbose output? (y/n): ")
ask = readline()
if ask == "y"
    verbosity=10
elseif ask == "n"
    verbosity=1
else
    error("Please give valid input (y/n).")
end

measure = [confusion_matrix, accuracy, bacc]
cv = CV(nfolds=6, shuffle=true)

ours = pdMISVMClassifier(C=1e3, μ=1e-5, ρ=1.2, exact=true)
ours_machine = machine(ours, X, y)
ours_eval = evaluate!(ours_machine, resampling=cv, measure=measure, verbosity=verbosity)
@show ours_eval

ours_inexact = pdMISVMClassifier(C=1e10, μ=1e-10, ρ=1.2, exact=false)
ours_inexact_machine = machine(ours_inexact, X, y)
ours_inexact_eval = evaluate!(ours_inexact_machine, resampling=cv, measure=measure, verbosity=verbosity)
@show ours_inexact_eval
