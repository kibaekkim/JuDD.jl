using Base: Bool
"""
Scenario and temporal decomposition of multistage stochastic MIP
"""

function multistage_decomposition!(
    graph::Plasmo.OptiGraph, 
    algo::LagrangeDual, 
    LM::AbstractLagrangeMaster,
    node_cluster::Vector{Vector{Plasmo.OptiNode}}
    )
    # convert node_clusters into set of models:: Dict{Int,JuMP.Model}
    # -> construct subgraph from set of nodes
    partition = Plasmo.Partition(graph, node_cluster)
    make_subgraphs!(partition)

    # -> convert Optigraph to JuMP.Model

    # for each models, add_block_model!(algo, blockid, models[blockid])

    coupling_variables = Vector{CouplingVariableRef}()
    # identify coupling variables -> how to convert linkconstraint to coupling variables

    # Set nonanticipativity variables as an array of symbols.
    set_coupling_variables!(algo, coupling_variables)

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    #run!(algo, LM)
    return algo
end

# ordinary scenario decomposition
function multistage_scenario_decomposition!(
    graph::Plasmo.OptiGraph, 
    algo::LagrangeDual, 
    LM::AbstractLagrangeMaster,
    )
    node_cluster = Vector{Vector{Plasmo.OptiNode}}()
    for node in Plasmo.getnodes(graph)
        if check_leaf(node)
            scenario = Vector{Plasmo.OptiNode}()
            current = node
            while true 
                pushfirst!(scenario, current)
                if isnothing(current.ext[:parent])
                    break
                else
                    current = current.ext[:parent]
                end
            end
            push!(node_cluster, scenario)
        end
    end
    multistage_decomposition!(graph, algo, LM, node_cluster)
end

# total temporal decomposition
function multistage_total_decomposition!(
    graph::Plasmo.OptiGraph, 
    algo::LagrangeDual, 
    LM::AbstractLagrangeMaster,
    )
    node_cluster = Vector{Vector{Plasmo.OptiNode}}()
    for node in Plasmo.getnodes(graph)
        if check_leaf(node)
            current = node
            while true 
                push!(node_cluster, [current])
                if isnothing(current.ext[:parent])
                    break
                else
                    current = current.ext[:parent]
                end
            end
        end
    end
    multistage_decomposition!(graph, algo, LM, node_cluster)
end
