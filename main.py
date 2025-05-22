from influence_analysis.graphs import GraphGenerator
from influence_analysis.propogation import IndependentCascadeModel
from influence_analysis.influence_algorithm import DegreeCentralityAlgorithm
from influence_analysis.simulator import Simulator
from tqdm import tqdm

graph = GraphGenerator.get_collab_graph()
prob = 0.3
num_timestep = 1000
num_seeds = 10
influence_algorithm = DegreeCentralityAlgorithm(graph, num_seeds)
prop_alg = IndependentCascadeModel(graph, prob)
simulator = Simulator(graph, influence_algorithm, prop_alg)

#simulator.seed_node('3466') # Seed one node manually
#simulator.seed_random_nodes(num_seeds) # Seed 5 random nodes
simulator.seed() # Seed according to the influence algorithm
print(f"Seeded Nodes: {simulator.get_active_nodes()}")

with tqdm(range(num_timestep), desc="Simulation Timestep") as t:
    print(f"Starting Simulation")
    for _ in t:
        simulator.timestep()
    print(f"Simulation Complete!")
    elapsed = t.format_dict['elapsed']
    print(f"Time: {elapsed:.4f} seconds ({elapsed/num_timestep:.4f} second per timestep)")
    print()

print(f"Post-Simulation Results after {num_timestep} time steps")
print(f"{simulator.get_num_active_nodes()} nodes activated")
print(f"Percent Active: {(simulator.get_num_active_nodes()/graph.number_of_nodes())*100:.2f}%")
print(simulator.get_active_nodes())