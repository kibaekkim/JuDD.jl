struct BnBNode
    LD::AbstractLagrangeDual
    lb::Float64
end

mutable struct BnBTree
    incumbent::Vector{BnBNode}     # list of nodes
    feasible_prob::Dict{Int,JuMP.Model}
    z_ub::Float64
    z_lb::Float64
    best_solution

    function BnBTree(LD::AbstractLagrangeDual)
        BT = new()

        root = BnBNode(LD,-Inf)
        BT.nodes = [root]
        BT.feasible_prob = get_problem_copy(LD)
        BT.z_ub = Inf
        BT.z_lb = -Inf
        best_solution = nothing

        return BT
    end
end



function get_problem_copy(LD::AbstractLagrangeDual)::Dict{Int,JuMP.Model}
    feasible_prob = Dict()
    for s in parallel.getpartition()
        m = copy(LD.block_model.model[s])
        #set_optimizer(m, Gurobi.Optimizer)
        #JuMP.set_optimizer_attribute(m, "OutputFlag", 0)

        feasible_prob[s] = m
    end
    return feasible_prob
end


function branch_and_bound(LD::AbstractLagrangeDual, LM::AbstractLagrangeMaster, initial_λ = nothing, time_limit::Float64)
    BnB = BnBTree(LD)

    tot_time = nothing
    if parallel.is_root()
        time_start = time()
        while true
            tot_time = time() - time_start
            if tot_time > time_limit
                println("maximum time limit reached")
                parallel.bcast(true)
                break
            elseif length(BnB.incumbent) == 0
                parallel.bcast(true)
                break
            end
            parallel.bcast(false)

            # take last element of incumbent
            # TODO: enhance node selection rule
            current_node = pop!(BnB.incumbent) 
            current = current_node.LD
            run!(current,LM,initial_λ,BnB.z_ub)

            # bounding
            bounding = current.block_model.dual_bound < BnB.z_ub
            parallel.bcast(bounding)
            if bounding                
                solution_identical, solution = solution_identical(current)
                parallel.bcast((solution_identical, solution))
                if solution_identical
                    println("solution identical!")
                    bound_by_solution(BnB, current, solution)
                else
                    println("solution not identical!")
                    # use heuristic to get feasible objective value and replace z_ub if it is smaller than z_ub
                    h_solution = heuristic_solution(solution)
                    parallel.bcast(h_solution)
                    bound_by_solution(BnB, current, h_solution)

                    #branching
                    z_dl = current.block_model.dual_bound
                    branching = z > z_dl * (1+1e-6)
                    parallel.bcast(branching)
                    if branching

                        parallel.bcast(z_dl)

                        # select one variable
                        # add bound constraints
                        next0, next1 = branch_variable(current, solution)

                        # push to incumbent
                        # always try increasing the lines
                        push!(BnB.incumbent, BnBNode(next0,z_dl))
                        push!(BnB.incumbent, BnBNode(next1,z_dl))
                    end
                end
                println("ub: ", BnB.z_ub, " lb: ", BnB.z_lb)
            end
            println("incumbent: ",length(BnB.incumbent))
        end
    else
        while true
            termination = parallel.bcast(nothing)
            if termination
                break
            end
            current_node = pop!(BnB.incumbent)
            current = current_node.LD
            run!(current,LM,initial_λ,BnB.z_ub)

            bounding = parallel.bcast(nothing)
            if bounding
                solution_identical(current)
                solution_identical, solution = parallel.bcast(nothing)

                if solution_identical
                    bound_by_solution(BnB, current, solution)
                else
                    h_solution = parallel.bcast(nothing)
                    bound_by_solution(BnB, current, h_solution)
                    branching = parallel.bcast(nothing)
                    if branching
                        z_dl = parallel.bcast(nothing)
                        next0, next1 = branch_variable(current, solution)

                        # push to incumbent
                        # always try increasing the lines
                        push!(BnB.incumbent, BnBNode(next0,z_dl))
                        push!(BnB.incumbent, BnBNode(next1,z_dl))
                    end
                end
            end
        end
    end

end

function bound_by_solution(BnB::BnBTree, LD::AbstractLagrangeDual, solution)
    # get feasible objective value and replace z_ub if it is smaller than z_ub
    if parallel.is_root()
        z = feasible_objective(BnB, solution)
        z_dl = LD.block_model.dual_bound
        if z < BnB.z_ub
            BnB.z_ub = z
            BnB.z_lb = z_dl
            BnB.best_solution = solution
            # delete all problems with dual bound greater than z_ub
            for p in length(BnB.incumbent):-1:1
                lb = BnB.incumbent[p].lb
                if lb > BnB.z_ub
                    deleteat!(BnB.incumbent,p)
                    parallel.bcast(p)
                end
            end
        end
        parallel.bcast(0)
    else
        feasible_objective(BnB, solution)
        delid = parallel.bcast(nothing)
        while delid != 0
            deleteat!(BnB.incumbent,p)
            delid = parallel.bcast(nothing)
        end
    end
end