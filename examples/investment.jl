using JuMP, Plasmo, GLPK, DualDecomposition

const DD = DualDecomposition

"""
a: interest rate
π: unit stock price
ρ: unit dividend price


K: number of stages
L: number of stock types
2^L scenarios in each stage
2^L^(K-1)=16 scenarios in total
ρ = 0.05 * π
bank: interest rate 0.01
stock1: 1.03 or 0.97
stock2: 1.06 or 0.94
...

b_k: initial asset (if k=1) and income (else)
B_k: money in bank
x_{k,l}: number of stocks to buy/sell (integer)
y_{k,l}: total stocks 

deterministic model:

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
const b_init = 100  # initial capital
const b_in = 30   # income

# iteratively add nodes
# root nde
function create_nodes!(graph::Plasmo.OptiGraph)
    nd = DD.add_node!(graph, ones(L))

    #subproblem formulation
    @variable(nd, x[l=1:L], Int)
    @variable(nd, y[l=1:L] >= 0)
    @variable(nd, B >= 0)
    π = nd.ext[:ξ]
    @constraints(nd, 
        begin
            B + sum( π[l] * x[l] for l in 1:L) == b_init
            [l=1:L], y[l] - x[l] == 0 
        end
    )
    @objective(nd, Max, nd.ext[:p] * 0)

    create_nodes!(graph, nd)
end
# child nodes
function create_nodes!(graph::Plasmo.OptiGraph, pt::Plasmo.OptiNode)
    for scenario = 1:2^L
        prob = 1/2^L
        ξ = get_realization(pt.ext[:ξ], scenario)
        nd = DD.add_node!(graph, ξ, pt, prob)

        #subproblem formulation
        @variable(nd, x[l=1:L], Int)
        @variable(nd, y[l=1:L] >= 0)
        @variable(nd, B >= 0)
        @variable(nd, y_[l=1:L] >= 0)
        @variable(nd, B_ >= 0)
        π = nd.ext[:ξ]
        ρ = pt.ext[:ξ] * 0.05
        @constraint(nd, B + sum( π[l] * x[l] - ρ[l] * y_[l] for l in 1:L) - (1+a) * B_ == b_in)
        @constraint(nd, [l=1:L], y[l] - x[l] - y_[l] == 0)

        @linkconstraint(graph, [l=1:L], nd[:y_][l] == pt[:y][l])
        @linkconstraint(graph, nd[:B_] == pt[:B])

        if nd.ext[:stage] < K
            @objective(nd, Max, nd.ext[:p] * 0)
            create_nodes!(graph, nd)
        else
            @constraint(nd, [l=1:L], x[l] == 0)
            @objective(nd, Max, nd.ext[:p] * (B + sum( π[l] * y[l] for l in 1:L )))
        end
    end
end

# construct realization event
function get_realization(ξ::Array{Float64,1}, scenario::Int)::Array{Float64,1}
    ret = ones(L)
    multipliers = digits(scenario - 1, base=2, pad=L)*2 - ones(L)
    for l = 1:L
        ret[l] = ξ[l] * (1 + multipliers[l] * l * 0.03)
    end
    return ret
end

# create graph
graph = Plasmo.OptiGraph()
create_nodes!(graph)
set_optimizer(graph,GLPK.Optimizer)
optimize!(graph)
println(objective_value(graph))