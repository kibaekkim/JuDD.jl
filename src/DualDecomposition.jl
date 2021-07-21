"""
    DualDecomposition

This module implements a block model for dual Decomposition and Lagrangian 
dual method for solving the block model. Applications of the dual Decomposition
include stochastic programming, temporal decomposition, and network decomposition.
"""
module DualDecomposition

using JuMP
using SparseArrays
using Printf
using LinearAlgebra
using BundleMethod
using Plasmo
const BM = BundleMethod

export BM

abstract type AbstractMethod end
abstract type AbstractLagrangeDual <: AbstractMethod end

"""
    run!

Empty function for `AbstractMethod`
"""
function run! end

include("parallel.jl")
include("BlockModel.jl")
include("LagrangeMaster/LagrangeMaster.jl")
include("LagrangeMaster/BundleMethod.jl")
include("LagrangeMaster/SubgradientMethod.jl")
include("LagrangeDual.jl")
include("ScenarioTree.jl")
include("utils.jl")

end  # module DualDecomposition
