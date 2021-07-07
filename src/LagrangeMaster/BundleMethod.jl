"""
    BundleMaster

Bundle method implementation of Lagrangian master method. 
This simply uses BundleMethod.jl package.

# Arguments
- `constructor`: struct for bundle method
- `optimizer`: optimization solver for bundle master
- `inner`: bundle method object
"""
mutable struct BundleMaster <: AbstractLagrangeMaster
    constructor::DataType
    optimizer::Union{Nothing,DataType,MOI.OptimizerWithAttributes}
    inner::Union{Nothing,BM.AbstractMethod}
    params::BM.Parameters

    function BundleMaster(constructor, optimizer, params = BM.Parameters())
        bm = new()
        bm.constructor = constructor
        bm.optimizer = optimizer
        bm.inner = nothing
        bm.params = params
        return bm
    end
end

function load!(method::BundleMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64}, bound::Union{Float64,Nothing})
    method.inner = method.constructor(num_coupling_variables, num_blocks, eval_function; init = init_sol, params = method.params)
    if !isnothing(bound)
        BM.set_obj_limit(method.inner, bound)
    end

    # Set optimizer to bundle method
    model = BM.get_jump_model(method.inner)
    JuMP.set_optimizer(model, method.optimizer)

    # This builds the bunlde model.
    BM.build_bundle_model!(method.inner)
end

function add_constraints!(LD::AbstractLagrangeDual, method::BundleMaster)
    model = BM.get_jump_model(method.inner)
    λ = model[:x]
    for (id, vars) in LD.block_model.variables_by_couple
        @constraint(model, sum(λ[index_of_λ(LD, v)] for v in vars) == 0)
    end
end

run!(method::BundleMaster) = BM.run!(method.inner)
get_objective(method::BundleMaster) = -BM.get_objective_value(method.inner)
get_solution(method::BundleMaster) = copy(BM.get_solution(method.inner))
get_times(method::BundleMaster)::Vector{Float64} = method.inner.model.time
