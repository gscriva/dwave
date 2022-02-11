# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:19:49 2021

@author: Benjamin
"""

import numpy as np
from numpy.random import rand
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
from tqdm import trange

from adjacency import Adjacency


def get_token():
    """Return your personal access token"""
    return "CINE-6bb0e25c6a6fafcaf548a48a27190c1694a5f762"


Lx = 22
N = Lx ** 2
np.random.seed(12345)
J = (np.random.uniform(-1., 1., size=(N - Lx, 2))) * -1.0
np.savetxt("coplings.txt", J)

Js = {}
hs = {}


def get_Js(J=J, Lx=Lx):
    for ky in range(Lx):
        for kx in range(Lx):

            k = kx + (Lx * ky)
            kR = k - ky  # coupling to the right of S0[kx,ky]
            kU = k - Lx  # coupling to the up of S0[kx,ky]
            kL = k - ky - 1  # coupling to the left of S0[kx,ky]
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
        #print(int(i+1), int(j+1), coupling)
        txtarr.append([int(i + 1), int(j + 1), coupling])
    np.savetxt(f"{Lx**2}spins_uniform-1nn.txt", txtarr, fmt=['%d', '%d', '%1.10f'])
    return Js


def econf(Lx, J, S0):
    energy = 0.0
    for kx in range(Lx):
        for ky in range(Lx):
            k = kx + (Lx * ky)
            kR = k - ky  # coupling to the right of S0[kx,ky]
            kD = k  # coupling to the down of S0[kx,ky]

            # Tries to find a spin to right, if no spin energy contribution is 0.
            Rs = S0[kx + 1, ky] * J[kR, 0] if (kx + 1) % Lx != 0 else 0
            # Tries to find a spin to left, if no spin energy contribution is 0.
            Ds = S0[kx, ky + 1] * J[kD, 1] if (ky + 1) % Lx != 0 else 0

            energy += -S0[kx, ky] * (Rs + Ds)
    return energy / (Lx ** 2)


def compute_energy_bis(sample, neighbours, couplings, len_neighbours):
    energy = 0
    for i in range(neighbours.shape[0]):
        for j in range(len_neighbours[i]):
            energy += sample[i] * (sample[int(neighbours[i, j])] * couplings[i, j])
    return energy / 2


def run_on_qpu(Js, hs, sampler, chain_strength):
    """Runs the QUBO problem Q on the sampler provided.

    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    sample_set = sampler.sample_ising(
        h=hs,
        J=Js,
        num_reads=numruns,
        label="ISING Glass Uniform Single NN",
        chain_strength=chain_strength,
        reduce_intersample_correlation=True,
        annealing_time=1,
        readout_thermalization=20,
        answer_mode="raw",
    )

    return sample_set


## ------- Main program -------
if __name__ == "__main__":

    numruns = 1000
    Js = get_Js()

    # bqm = dimod.BQM.from_qubo(Js)
    # sample_set = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=numruns)

    qpu = DWaveSampler(solver={"topology__type": "pegasus"})

    sampler = EmbeddingComposite(qpu)

    txtfile = "484spins_uniform-1nn.txt"
    spin_side = 22
    open_bound_couplings = Adjacency(spin_side)
    open_bound_couplings.loadtxt(txtfile)

    # get neighbourhood matrix
    neighbours, couplings = open_bound_couplings.get_neighbours()
    neighbours = neighbours.astype(int)
    len_neighbours = np.sum(couplings != 0, axis=-1)

    #print(sampler.properties)

    #chain_strengths = np.linspace(0.25, 1., num=4)
    #print(chain_strengths)
    #print(np.loadtxt(txtfile).max(0)[2], np.loadtxt(txtfile).min(0)[2])

    for k in trange(10, 11):
        sample_set = run_on_qpu(Js, hs, sampler, 1.1)

        print(f"K={k}\n", sample_set)

        dwave.inspector.show(sample_set)

        configs = []
        energies = []
        energies_bis = []
        dwave_engs = []

        for i in range(sample_set.record.size):
            for j in range(sample_set.record[i][2]):

                S0 = sample_set._record[i]["sample"]
                dwave_engs.append(sample_set._record[i]["energy"])

                # S0d = np.reshape(S0, (Lx, Lx), order="F")
                # s0 = np.reshape(S0d, Lx**2)
                #energy = econf(Lx, J, S0d)
                #energy_bis = compute_energy_bis(S0, neighbours, couplings, len_neighbours)

                configs.append(S0)
                #energies.append(energy)
                #energies_bis.append(energy_bis)

                #print(f"{sample_set._record[i]['energy'] / spin_side**2}   {energy_bis / spin_side**2}")


        #np.save(f"484spins-1nn-uniform-100mus/configs_{k}.npy", np.asarray(configs))
        #np.save(f"484spins-1nn-uniform-100mus/dwave-engs_{k}.npy", np.asarray(dwave_engs))

        print(np.asarray(dwave_engs).min() / 484, np.asarray(dwave_engs).mean() / 484)
        #np.savetxt(f"energies_{k}.txt", energies)
        #np.savetxt(f"energies_{k}_bis.txt", energies_bis)

