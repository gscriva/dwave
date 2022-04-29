import itertools
import pickle
from datetime import datetime

import dwave_networkx as dnx
import numpy as np
from minorminer import find_embedding

import dwave.inspector
from adjacency import Adjacency
from dwave.system import DWaveSampler, FixedEmbeddingComposite


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


if __name__ == "__main__":
    print("\nSearching for an embedding...")
    start_time = datetime.now()


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
    topology = "pegasus"

    # embeddidng parameters
    timeout = 3600
    max_no_improvement = 100
    tries = 50
    chainlength_patience = 100

    dwave_sampler = DWaveSampler(solver={'topology__type': topology})
    saved_chain_len, saved_qubits = np.inf, np.inf
    while True:
        # find embedding
        embedding = find_embedding(ising_graph, dwave_sampler.edgelist, timeout=timeout, 
                                max_no_improvement=max_no_improvement, tries=tries,
                                chainlength_patience=chainlength_patience)
        max_chain_len = 1
        tot_qubits = 0
        for qubits in embedding.values():
            tot_qubits += len(qubits) 
            if len(qubits) < max_chain_len:
                continue
            max_chain_len = len(qubits)
        print(f"Maximum chain length {max_chain_len} [{saved_chain_len}] Total qubits {tot_qubits} [{saved_qubits}]\n")
        if max_chain_len < saved_chain_len or tot_qubits < saved_qubits: 
            with open(f'embedding_{spins}spins_{connectivity}nn.pkl', 'wb') as f:
                pickle.dump(embedding, f)
            saved_chain_len, saved_qubits = max_chain_len, tot_qubits

    print(f"Duration {datetime.now() - start_time}")
