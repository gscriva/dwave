import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
from tqdm import trange

from adjacency import Adjacency


def get_token():
    """Return your personal access token"""
    return "CINE-6bb0e25c6a6fafcaf548a48a27190c1694a5f762"


def get_Js(J, Lx):
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
    txtarr = []
    for (i, j), coupling in Js.items():
        # see http://mcsparse.uni-bonn.de/spinglass/
        txtarr.append([int(i + 1), int(j + 1), coupling])
    np.savetxt(f"{Lx**2}spins-uniform-1nn.txt", txtarr, fmt=['%d', '%d', '%1.10f'])
    return Js

def compute_energy_bis(sample, neighbours, couplings, len_neighbours):
    energy = 0
    for i in range(neighbours.shape[0]):
        for j in range(len_neighbours[i]):
            energy += sample[i] * (sample[int(neighbours[i, j])] * couplings[i, j])
    return energy / 2

    
def run_on_qpu(Js, 
               hs, 
               sampler,
               num_reads=1000, 
               label="ISING Uniform 1NN", 
               chain_strength=None, 
               annealing_time=1
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
        annealing_time=annealing_time,
        answer_mode="raw",
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
    txtfile = f"{spins}spins-uniform-1nn"
    np.savetxt(f"{txtfile}.txt", J)
    Js = get_Js(J, spin_side)
    # set dwave properties
    num_reads = 3500
    annealing_time = 200
    topology = "pegasus"

    # initialize sampler
    qpu = DWaveSampler(solver={"topology__type": topology})
    sampler = EmbeddingComposite(qpu)
    # use custom class for couplings 
    open_bound_couplings = Adjacency(spin_side)
    open_bound_couplings.loadtxt(f"{txtfile}.txt")
    # get neighbourhood matrix
    neighbours, couplings = open_bound_couplings.get_neighbours()
    neighbours = neighbours.astype(int)
    len_neighbours = np.sum(couplings != 0, axis=-1)

    for k in range(50, 54):
        sample_set = run_on_qpu(Js, 
                                {}, 
                                sampler, 
                                num_reads=num_reads, 
                                label="ISING Uniform 1NN", 
                                chain_strength=1.1, 
                                annealing_time=annealing_time)
        
        #print(f"K={k}\n", sample_set)
        #dwave.inspector.show(sample_set)

        configs = []
        dwave_engs = []
        for i in range(sample_set.record.size):
            for j in range(sample_set.record[i][2]):
                configs.append(sample_set._record[i]["sample"])
                dwave_engs.append(sample_set._record[i]["energy"])

        np.save(f"{txtfile}-100mus/configs_{k}", np.asarray(configs))
        np.save(f"{txtfile}-100mus/dwave-engs_{k}.npy", np.asarray(dwave_engs))

        print(np.asarray(dwave_engs).min() / 484, np.asarray(dwave_engs).mean() / 484)