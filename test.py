import dwavebinarycsp as dbc 
import dwavebinarycsp.factories.constraint.gates as gates 
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite 
from dwave.system.samplers import DWaveSampler 
from minorminer import find_embedding

 
# # Making two different BQMs (think: energy functions or optimization functions) 
# csp1 = dbc.ConstraintSatisfactionProblem(dbc.BINARY) 
# csp1.add_constraint(gates.and_gate(['a','b','c'])) 
# bqm1 = dbc.stitch(csp1) 
 
# csp2 = dbc.ConstraintSatisfactionProblem(dbc.BINARY) 
# csp2.add_constraint(gates.or_gate(['a','b','c'])) 
# bqm2 = dbc.stitch(csp2) 
 
# # Using Embedding Composite 
# sampler = EmbeddingComposite(DWaveSampler()) 
# sampler.sample(bqm1) # Gets a new embedding for bqm1 
# sampler.sample(bqm2) # Gets a new embedding for bqm2 
 
# # Using Fixed Embedding Composite 
# # Note: bqm1 and bqm2 can both be represented by the same graph - triangle graph. 
# embedding = {'a':[0,4],'b':[1],'c':[5]} # Embedding the triangle graph using QPU indices
# fixedSampler = FixedEmbeddingComposite(DWaveSampler(), embedding) 
# fixedSampler.sample(bqm1) 
# fixedSampler.sample(bqm2) # in both samples, the SAME embedding gets used

import dimod
import dwave_networkx as dnx
from dwave.system import FixedEmbeddingComposite



c1 = [(0,1), (0,2), (1,0), (2,0)]
dwave_sampler = DWaveSampler()
print(dwave_sampler.edgelist[:10])

embedding = find_embedding(c1, dwave_sampler.edgelist)
print(embedding)
# structured_sampler = dimod.StructureComposite(dimod.ExactSolver(),
#                                                c1.nodes, c1.edges)
sampler = FixedEmbeddingComposite(dwave_sampler, embedding)
print(sampler.target_structure)
sample_set = sampler.sample_ising(h={}, J={(0,1): -1, (0,2): -1})

for record in sample_set.record:
    print(record['sample'])
    print(record['energy'])


#[('a', 'b'), ('a', 'c'), ('b', 'c')]