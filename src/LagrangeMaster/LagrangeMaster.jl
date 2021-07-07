"""
    AbstractLagrangeMaster

Abstract type of Lagrangian master methods
"""
abstract type AbstractLagrangeMaster end

"""
    load!

This function loads the Lagrangian dual problem to the master.

# Arguments
- `method`: Lagrangian master method
- `num_coupling_variables`: number of coupling variables
- `num_blocks`: number of blocks (or Lagrangian subproblems)
- `eval_function`: function pointer to evalute Lagrangian dual functions
- `init_sol`: initial solution of the Lagrangian master
"""
function load!(method::AbstractLagrangeMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64}, bound::Union{Float64,Nothing}) end

"""
    add_constraints!

This function is to add any constraints to the master problem.

# Arguments
- `LD`: Lagrangian dual
- `method`: Lagrangian master
"""
function add_constraints!(LD::AbstractLagrangeDual, method::AbstractLagrangeMaster) end


"""
    run!

This runs the Lagrangian master method.

# Arguments
- `method`: Lagrangian master
"""
function run!(method::AbstractLagrangeMaster) end


"""
    get_objective

This returns the objective function value.

# Arguments
- `method`: Lagrangian master
"""
function get_objective(method::AbstractLagrangeMaster) end


"""
    get_solution

This returns the Lagrangian master solution.

# Arguments
- `method`: Lagrangian master
"""
function get_solution(method::AbstractLagrangeMaster) end


"""
    get_times

This returns the solution times of Lagrangian master for all iterations.

# Arguments
- `method`: Lagrangian master
"""
get_times(method::AbstractLagrangeMaster)::Vector{Float64} = zeros(1)