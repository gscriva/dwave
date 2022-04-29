import itertools
import pickle

import numpy as np
from tqdm import trange

import dwave.inspector
from adjacency import Adjacency
from dwave.system import DWaveSampler, FixedEmbeddingComposite


def get_token():
    """Return your personal access token"""
    #return "CINE-6bb0e25c6a6fafcaf548a48a27190c1694a5f762"
    return "DEV-ed754d76dd0318480f2c1ba2747bfa8d946c9ae8"

def get_Js(J, Lx, J2=None):
    connectivity=1 if J2 is None else 3
    Js={}
    for ky in range(Lx):
        for kx in range(Lx):

            k = kx + (Lx * ky)
            kR = k - ky  # coupling to the right of S0[kx,ky]
            kD = k  # coupling to the down of S0[kx,ky]
            D = k + Lx
            R = k + 1
            if k < ((Lx * ky) + (Lx - 1)):
                JR = J[int(kR), 0] * 1.0
                Js.update({(k, R): JR})
                #print(k)
            if k < (Lx ** 2 - 1) - Lx + 1:
                JD = J[int(kD), 1] * 1.0
                Js.update({(k, D): JD})
                #print(k)

            if J2 is not None:
                UR = k - (Lx - 1)  # Spin up right
                DR = k + (Lx + 1)  # spin down right
                kUR = k - Lx - (ky - 1)  # coupling up right
                kDR = k - (ky)  # coupling down right

                if k < (Lx * (Lx - 1)) and kx != Lx - 1:
                    JDR = J2[int(kDR), 1] * 1.0
                    Js.update({(k, DR): JDR})

                if ky > 0 and kx != Lx - 1:
                    JUR = J2[int(kUR), 0] * 1.0
                    Js.update({(k, UR): JUR})
    # .sg file is for the spin glass server 
    f = open(f"{Lx**2}spins-uniform-{connectivity}nn.sg", 'w')
    f.write(f"{Lx*Lx} {len(Js)}\n")
    couplings_txt = []
    ising_graph = []
    for (i, j), coupling in Js.items():
        # see http://mcsparse.uni-bonn.de/spinglass/
        # spin glass server minimize -H
        couplings_txt.append([int(i + 1), int(j + 1), coupling])
        ising_graph.append((i,j))
        f.write(f"{int(i + 1)} {int(j + 1)} {-coupling}\n")

    np.savetxt(f"{Lx**2}spins-uniform-{connectivity}nn.txt", couplings_txt, fmt=['%d', '%d', '%1.10f'])
    return Js, ising_graph

def compute_energy(sample, neighbours, couplings, len_neighbours):
    energy = 0
    for i in range(neighbours.shape[0]):
        for j in range(len_neighbours[i]):
            energy += sample[i] * (sample[int(neighbours[i, j])] * couplings[i, j])
    return energy / 2

    
def run_on_qpu(Js, 
               hs, 
               sampler,
               num_reads=1000, 
               label="ISING Uniform 1NN Reduced", 
               chain_strength=None, 
               anneal_schedule=None,
               ):
    """Runs the QUBO problem Q on the sampler provided."""

    sample_set = sampler.sample_ising(
        h=hs,
        J=Js,
        num_reads=num_reads,
        label=label,
        chain_strength=chain_strength,
        reduce_intersample_correlation=True,
        readout_thermalization=50,
        anneal_schedule=anneal_schedule,
        #annealing_time=10,
        answer_mode="raw",
        return_embedding=True,
    )

    return sample_set


## ------- Main program -------
if __name__ == "__main__":
    # set parameters
    spin_side = 22
    spins = spin_side ** 2

    # create couplings and save
    np.random.seed(12345)
    J = (np.random.uniform(-1., 1., size=(spins - spin_side, 2))) * -1.0
    J2 = None
    #J2 = (np.random.uniform(-1.0, 1.0, size=((spins - spin_side) ** 2, 2))) * -1.0
    connectivity = 1 if J2 is None else 3

    txtfile = f"{spins}spins-uniform-{connectivity}nn"
    Js, ising_graph = get_Js(J, spin_side, J2)

    # set dwave properties
    num_reads = 8000
    anneal_schedule = [[0.0,0.0],[9.3,0.7],[10.,1.0]]
    topology = "pegasus"

    # load embeddding
    with open(f'embedding_{spins}spins_{connectivity}nn.pkl', 'rb') as f:
        embedding = pickle.load(f)

    # initialize sampler
    qpu = DWaveSampler(solver={"topology__type": topology}, token=get_token())
    sampler = FixedEmbeddingComposite(qpu, embedding)
    #sampler = EmbeddingComposite(qpu)

    # use custom class for couplings 
    open_bound_couplings = Adjacency(spin_side)
    open_bound_couplings.loadtxt(f"{txtfile}.txt")

    # get neighbourhood matrix
    neighbours, couplings = open_bound_couplings.get_neighbours()
    neighbours = neighbours.astype(int)
    len_neighbours = np.sum(couplings != 0, axis=-1)

    #chain_strs = np.linspace(0.25, 4., 16)
    for k in range(6, 7):
        sample_set = run_on_qpu(Js, 
                                {}, 
                                sampler, 
                                num_reads=num_reads, 
                                label="ISING Uniform 3NN", 
                                chain_strength=1.15, 
                                anneal_schedule=anneal_schedule)
        
        # print(f"smaple.info {sample_set.info['embedding_context']}")
        # dwave.inspector.show(sample_set)

        configs = []
        dwave_engs = []
        for i in range(sample_set.record.size):
            for j in range(sample_set.record[i][2]):
                configs.append(sample_set._record[i]["sample"])
                dwave_engs.append(sample_set._record[i]["energy"])

                #print(f"eng {compute_energy(configs[-1], neighbours, couplings, len_neighbours)} dwave_eng {dwave_engs[-1]}")

        print(f"Block {k} {np.asarray(dwave_engs).min() / 484} {np.asarray(dwave_engs).mean() / 484}")

        np.save(f"{txtfile}-{int(anneal_schedule[-1][0])}mus/configs_{k}", np.asarray(configs))
        np.save(f"{txtfile}-{int(anneal_schedule[-1][0])}mus/dwave-engs_{k}.npy", np.asarray(dwave_engs))
