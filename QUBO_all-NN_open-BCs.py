# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:19:49 2021
@author: Benjamin
"""

import numpy as np
from numpy.random import rand
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
import dimod

from adjacency import Adjacency



def get_token():
    """Return your personal access token"""
    return "CINE-6bb0e25c6a6fafcaf548a48a27190c1694a5f762"


Lx = 22
N = Lx ** 2
np.random.seed(12345)
J = (np.random.uniform(-1.0, 1.0, size=(N - Lx, 2))) * -1.0
J2 = (np.random.uniform(-1.0, 1.0, size=((Lx - 1) ** 2, 2))) * -1.0
# J = (np.random.normal(0.0, 1.0, size=(N - Lx, 2))) * -1.0
#J2 = (np.random.normal(0.0, 1.0, size=((Lx - 1) ** 2, 2))) * -1.0
np.savetxt("coplings.txt", J)

Js = {}
hs = {}

def compute_energy_bis(sample, neighbours, couplings, len_neighbours):
    energy = 0
    for i in range(neighbours.shape[0]):
        for j in range(len_neighbours[i]):
            energy += sample[i] * (sample[int(neighbours[i, j])] * couplings[i, j])
    return energy / 2

def get_Js(J=J, Lx=Lx):
    for ky in range(Lx):
        for kx in range(Lx):

            k = kx + (Lx * ky)
            kR = k - ky  # coupling to the right of S0[kx,ky]
            kU = k - Lx  # coupling to the up of S0[kx,ky]
            kL = k - ky - 1  # coupling to the left of S0[kx,ky]
            kD = k  # coupling to the down of S0[kx,ky]

            D = k + Lx  # Spin down
            R = k + 1  # Spin right

            if k < ((Lx * ky) + (Lx - 1)):
                JR = J[int(kR), 0] * 1.0
                Js.update({(k, R): JR})

            if k < (Lx ** 2 - 1) - Lx + 1:
                JD = J[int(kD), 1] * 1.0
                Js.update({(k, D): JD})

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
    txtarr = []
    for (i, j), coupling in Js.items():
        # see http://mcsparse.uni-bonn.de/spinglass/
        #print(int(i+1), int(j+1), coupling)
        txtarr.append([int(i + 1), int(j + 1), coupling])
    np.savetxt(f"{Lx**2}spins_open-3nn.txt", txtarr, fmt=['%d', '%d', '%1.10f'])

    return Js


def econf(Lx, J, S0):
    energy = 0.0
    rs_count = 0
    ds_count = 0
    urs_count = 0
    drs_count = 0
    for kx in range(Lx):
        for ky in range(Lx):

            k = kx + (Lx * ky)
            R = kx + 1  # right spin
            D = ky + 1  # down spin
            DR = k + Lx + 1
            UR = k - (Lx - 1)

            kR = k - ky  # coupling to the right of S0[kx,ky]
            kD = k  # coupling to the down of S0[kx,ky]
            kR = k - ky  # coupling to the right of S0[kx,ky]
            kD = k  # coupling to the down of S0[kx,ky]

            kUR = k - Lx - (ky - 1)  # coupling up right
            kDR = k - (ky)  # coupling down right

            try:
                Rs = S0[R, ky] * J[kR, 0]
                rs_count += (
                    1  # Tries to find a spin to right, if no spin, contribution is 0.
                )

            except:
                Rs = 0.0

            try:
                Ds = (
                    S0[kx, D] * J[kD, 1]
                )  # Tries to find a spin down, if no spin, contribution is 0.
                ds_count += 1
            except:
                Ds = 0.0

            if ky > 0 and kx != Lx - 1:
                URs = (
                    S0[kx + 1, ky - 1] * J2[kUR, 0]
                )  # Tries to find a spin to right, if no spin, contribution is 0.
                urs_count += 1

            else:
                URs = 0.0

            try:
                DRs = (
                    S0[kx + 1, ky + 1] * J2[kDR, 1]
                )  # Tries to find a spin down, if no spin, contribution is 0.
                drs_count += 1
            except:
                DRs = 0.0

            nb = Rs + Ds + URs + DRs  # + Ls + Us
            S = S0[kx, ky]
            energy += -S * nb
    #print("rs", rs_count, "ds", ds_count, "urs", urs_count, "drs", drs_count)
    return energy / (Lx ** 2)


def run_on_qpu(Js, hs, sampler):
    """Runs the QUBO problem Q on the sampler provided.
    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    print(numruns)

    sample_set = sampler.sample_ising(
        h=hs,
        J=Js,
        num_reads=numruns,
        label="ISING Glass open BCs",
        reduce_intersample_correlation=True,
        annealing_time=1,
        answer_mode="raw",
    )

    return sample_set


## ------- Main program -------
if __name__ == "__main__":

    numruns = 10
    Js = get_Js()

    # bqm = dimod.BQM.from_qubo(Js)
    # sample_set = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=numruns)

    qpu_2000q = DWaveSampler(solver={"topology__type": "pegasus"})

    sampler = EmbeddingComposite(qpu_2000q)

    txtfile = "484spins_open-3nn.txt"
    open_bound_couplings = Adjacency(Lx)
    open_bound_couplings.loadtxt(txtfile)

    # get neighbourhood matrix
    neighbours, couplings = open_bound_couplings.get_neighbours()
    neighbours = neighbours.astype(int)
    len_neighbours = np.sum(couplings != 0, axis=-1)

    #print(sampler.properties)

    for k in range(0,1):
        sample_set = run_on_qpu(Js, hs, sampler)

        print(sample_set)
        #import dwave.inspector
        #dwave.inspector.show(sample_set)
        #break
        configs = []
        energies = []
        energies_bis = []
        dwave_engs = []

        for i in range(sample_set.record.size):
            for j in range(sample_set.record[i][2]):

                S0 = sample_set._record[i]["sample"]
                dwave_engs.append(sample_set._record[i]["energy"])


                S0d = np.reshape(S0, (Lx, Lx), order="F")

                energy = econf(Lx, J, S0d)
                energy_bis = compute_energy_bis(S0, neighbours, couplings, len_neighbours)

                print(f"engs {dwave_engs[i] / Lx**2 :.20f} {-energy: .20f} {energy_bis / Lx**2: .20f}")

                configs.append(S0)
                energies.append(dwave_engs)

        np.save("configs" + str(k) + ".npy", np.asarray(configs))
        np.savetxt(f"dwave-engs_{k}.txt", dwave_engs)
        # np.savetxt(f"energies_{k}.txt", energies)
        # np.savetxt(f"energies_{k}_bis.txt", energies_bis)
        # np.savetxt("energies" + str(k) + ".txt", energies)


dwave.inspector.show(sample_set)
