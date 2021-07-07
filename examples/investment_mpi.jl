using JuMP, Ipopt, GLPK
using DualDecomposition
using Random

const DD = DualDecomposition


"""
a: interest rate
π: unit stock price
ρ: unit dividend price
K = 3 number of stages
L = 2 number of investment vehicles
2^L scenarios in each stage
2^L^(K-1)=16 scenarios in total
ρ = 0.05 * π
bank: interest rate 0.01
asset1: 1.03 or 0.97
asset2: 1.06 or 0.94
"""

function create_tree(K::Int, L::Int)::DD.Tree
    π = ones(L)                              
    tree = DD.Tree(π)
    add_nodes!(K, L, tree, 1, 1)
    return tree
end

function add_nodes!(K::Int, L::Int, tree::DD.Tree, id::Int, k::Int)
    if k < K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            DD.addchild!(tree, id, π)
            childid = length(tree.nodes)
            add_nodes!(K, L, tree, childid, k+1)
        end
    elseif k == K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            DD.addchild!(tree, id, π)
        end
    end
end

function iterlist(L::Int, π::Array{Float64})::Array{Array{Float64}}
    # generates all combinations of up and down scenarios
    ls = [Float64[] for _ in 1:2^L]
    ii = 1

    function foo(L::Int, l::Int, arr::Vector{Float64})
        up = (1.0 + 0.03 * l) * π[l]
        dn = (1.0 - 0.03 * l) * π[l]

        if l < L
            arr1 = copy(arr)
            arr1[l] = up
            foo(L, l+1, arr1)

            arr2 = copy(arr)
            arr2[l] = dn
            foo(L, l+1, arr2)
        else
            arr1 = copy(arr)
            arr1[l] = up
            ls[ii] = arr1
            ii+=1

            arr2 = copy(arr)
            arr2[l] = dn
            ls[ii] = arr2
            ii+=1
        end
    end

    foo(L, 1, Array{Float64}(undef, L))
    return ls
end

"""
    max     B_K+∑_{l=1}^{L}π_{K,l}y_{K,l}
    s.t.    B_1+∑_{l=1}^{L}π_{1,l}x_{1,l} = b_1
            b_k+(1+a)B_{k-1}+∑_{l=1}^{L}ρ_{k,l}y_{k-1,l} = B_k+∑_{l=1}^{L}π_{k,l}x_{k,l}, ∀ k=2,…,K
    
            y_{1,l} = x_{1,l}, ∀ l=1,…,L
    
            y_{k-1,l}+x_{k,l} = y_{k,l}, ∀ k=2,…,K, l=1,…,L
    
            x_{k,l} ∈ ℤ , ∀ k=1,…,K, l=1,…,L
    
            y_{k,l} ≥ 0, ∀ k=1,…,K, l=1,…,L
    
            B_k ≥ 0, ∀ k=1,…,K.
"""
const K = 3
const L = 2
const a = 0.01
const b1 = 100  # initial capital
const b2 = 30   # income

function create_scenario_model(K::Int, L::Int, tree::DD.Tree, id::Int)
    hist = DD.get_history(tree, id)
    m = Model(GLPK.Optimizer) 
    #@variable(m, x[1:K,1:L], integer=true)
    @variable(m, x[1:K,1:L])
    @variable(m, y[1:K,1:L]>=0)
    @variable(m, B[1:K]>=0)

    π = tree.nodes[1].ξ

    @constraint(m, B[1] + sum( π[l] * x[1,l] for l in 1:L) == b1)

    for l in 1:L
        @constraint(m, y[1,l]-x[1,l]==0)
    end

    for k = 2:K
        π = tree.nodes[hist[k]].ξ
        ρ = tree.nodes[hist[k-1]].ξ * 0.05

        @constraint(m, B[k] + sum( π[l] * x[k,l] - ρ[l] * y[k-1,l] for l in 1:L)
            - (1+a) * B[k-1] == b2)
        for l in 1:L
            @constraint(m, y[k,l]-x[k,l]-y[k-1,l]==0)
        end
    end
    for l in 1:L
        @constraint(m, x[K,l]==0)
    end
    π = tree.nodes[id].ξ
    @objective(m, Min, - (B[K] + sum( π[l] * y[K,l] for l in 1:L ))/(2^L)^(K-1) )
    return m
end

"""
The main computation section
"""
function main_comp()
    # generate tree data structure
    tree = create_tree(K,L)

    # Initialize MPI
    DD.parallel.init()

    # Create DualDecomposition instance.
    algo = DD.LagrangeDual()
        
    # Lagrange master method
    LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

    # compute dual decomposition method
    dual_decomp!(L, tree, algo, LM)
end


function dual_decomp!(L::Int, tree::DD.Tree, algo::DD.LagrangeDual, LM::DD.AbstractLagrangeMaster)

    nodelist = DD.get_stage_id(tree)
    leafdict = DD.leaf2block(nodelist[K])

    # partition scenarios into processes
    DD.parallel.partition(length(nodelist[K]))

    # Add Lagrange dual problem for each scenario s.
    models = Dict{Int,JuMP.Model}(nodelist[K][s] => create_scenario_model(K,L,tree,nodelist[K][s]) for s in DD.parallel.getpartition())
    for s in DD.parallel.getpartition()
        id = nodelist[K][s]
        DD.add_block_model!(algo, s, models[id])
    end

    coupling_variables = Vector{DD.CouplingVariableRef}()
    for s in DD.parallel.getpartition()
        id = nodelist[K][s]
        model = models[id]
        hist = DD.get_history(tree, id)
        yref = model[:y]
        Bref = model[:B]
        for k in 1:K-1
            root = hist[k]
            for l in 1:L
                push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, l], yref[k, l]))
            end
            push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, L+1], Bref[k]))
        end
        # dummy coupling variables
        for l in 1:L
            push!(coupling_variables, DD.CouplingVariableRef(s, [id, l], yref[K, l]))
        end
        Bref = model[:B]
        push!(coupling_variables, DD.CouplingVariableRef(s, [id, L+1], Bref[K]))
    end

    # Set nonanticipativity variables as an array of symbols.
    DD.set_coupling_variables!(algo, coupling_variables)

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    DD.run!(algo, LM)
end