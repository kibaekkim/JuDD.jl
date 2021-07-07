"""
    SubgradientMaster

Implementation of projected subgradient method
"""
mutable struct SubgradientMaster <: AbstractLagrangeMaster
    num_vars::Int
    num_functions::Int
    eval_f::Union{Nothing,Function}

    iter::Int # current iteration count
    maxiter::Int
    obj_limit::Float64

    f::Float64
    best_f::Float64
    x::Vector{Float64}
    best_x::Vector{Float64}
    ∇f::Vector{Float64}
    α::Float64 # step size

    step_size::Function

    constraint_matrix::Union{Nothing,SparseMatrixCSC{Float64,Int}}

    iteration_time::Vector{Float64}

    function SubgradientMaster()
        sg = new()
        sg.num_vars = 0
        sg.num_functions = 0
        sg.eval_f = nothing
        sg.iter = 0
        sg.maxiter = 1000
        sg.obj_limit = -Inf
        sg.f = -Inf
        sg.best_f = -Inf
        sg.x = []
        sg.best_x = []
        sg.∇f = []
        sg.α = 0.1
        # sg.step_size = (method) -> method.α / method.iter # square summable but not summable
        sg.step_size = (method) -> method.α / sqrt(method.iter) # Nonsummable diminishing
        sg.constraint_matrix = nothing
        sg.iteration_time = []
        return sg
    end
end

function load!(method::SubgradientMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64}, bound::Union{Float64,Nothing})
    method.num_vars = num_coupling_variables
    method.num_functions = num_blocks
    method.eval_f = eval_function
    method.x = init_sol
    method.best_x = init_sol
    if !isnothing(bound)
        method.obj_limit = bound
        #TODO: implement termination function
    end
end

function add_constraints!(LD::AbstractLagrangeDual, method::SubgradientMaster)
    I = Int[]; J = Int[]; V = Float64[]
    row = 1
    for (id, vars) in LD.block_model.variables_by_couple
        for v in vars
            push!(I, row)
            push!(J, index_of_λ(LD, v))
            push!(V, 1.0)
        end
        row += 1
    end
    method.constraint_matrix = sparse(I, J, V)
end

function run!(method::SubgradientMaster)

    # projection matrix
    m, n = size(method.constraint_matrix)
    I = sparse(SparseArrays.I, n, n)
    A = method.constraint_matrix
    proj = transpose(A) * ((A * transpose(A)) \ A)

    total_time = 0.0
    total_master_time = 0.0
    total_stime = time()

    method.∇f = zeros(method.num_vars)

    while method.iter < method.maxiter
        method.iter += 1
        if method.iter % 100 == 1
            @printf("%6s", "Iter")
            @printf("\t%13s", "fbest")
            @printf("\t%13s", "f")
            @printf("\t%13s", "|∇f|")
            @printf("\t%13s", "α")
            # @printf("\t%13s", "res")
            @printf("\t%7s", "sg time")
            @printf("\t%8s", "tot time")
            @printf("\n")
        end
        f, g = method.eval_f(method.x)
        method.f = -sum(f)
        if method.best_f < method.f
            method.best_f = method.f
            copy!(method.best_x, method.x)
        end

        master_stime = time()

        fill!(method.∇f, 0.0)
        for (k, gk) in g
            method.∇f .+= gk
        end

        # update
        α = method.step_size(method)
        method.x .-= α * method.∇f

        # projection
        method.x .-= proj * method.x
        # res = method.constraint_matrix * method.x

        # clock iteration time
        push!(method.iteration_time, time() - master_stime)
        total_master_time += time() - master_stime
        total_time = time() - total_stime

        @printf("%6d", method.iter)
        @printf("\t%+6e", method.best_f)
        @printf("\t%+6e", method.f)
        @printf("\t%+6e", norm(method.∇f))
        @printf("\t%+6e", α)
        # @printf("\t%+6e", norm(res))
        @printf("\t%7.2f", total_master_time)
        @printf("\t%8.2f", total_time)
        @printf("\n")
    end
end

get_objective(method::SubgradientMaster) = method.best_f
get_solution(method::SubgradientMaster) = method.best_x
get_times(method::SubgradientMaster)::Vector{Float64} = method.iteration_time
