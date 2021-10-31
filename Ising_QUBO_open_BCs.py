# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:19:49 2021

@author: Benjamin
"""

import numpy as np
from numpy.random import rand
from dwave.system import DWaveSampler, EmbeddingComposite


def get_token():
    """Return your personal access token"""
    return "CINE-6bb0e25c6a6fafcaf548a48a27190c1694a5f762"


Lx = 10
N = Lx ** 2
np.random.seed(12345)
J = (np.random.normal(0.0, 1.0, size=(N - Lx, 2))) * -1.0
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
                print(k)

            if k < (Lx ** 2 - 1) - Lx + 1:
                JD = J[int(kD), 1] * 1.0
                Js.update({(k, D): JD})
                print(k)

    return Js


def econf(Lx, J, S0):
    energy = 0.0
    for kx in range(Lx):
        for ky in range(Lx):

            k = kx + (Lx * ky)
            R = kx + 1  # right spin
            L = kx - 1  # left spin
            U = ky - 1  # up spin
            D = ky + 1  # down spin

            kR = k - ky  # coupling to the right of S0[kx,ky]
            kU = k - Lx  # coupling to the up of S0[kx,ky]
            kL = k - ky - 1  # coupling to the left of S0[kx,ky]
            kD = k  # coupling to the down of S0[kx,ky]

            try:
                Rs = (
                    S0[R, ky] * J[kR, 0]
                )  # Tries to find a spin to right, if no spin, contribution is 0.
            except:
                Rs = 0.0

            try:
                Ds = S0[kx, D] * J[kD, 1]
            except:
                Ds = 0.0

            nb = Rs + Ds  # + Ls + Us
            S = S0[kx, ky]
            energy += -S * nb
    return energy / (Lx ** 2)


def run_on_qpu(Js, hs, sampler):
    """Runs the QUBO problem Q on the sampler provided.

    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    sample_set = sampler.sample_ising(
        h=hs,
        J=Js,
        num_reads=numruns,
        label="ISING Glass open BCs Single NN",
        reduce_intersample_correlation=True,
        # programming_thermalization=2,
        annealing_time=10,
        # readout_thermalization=2,
        # postprocess="sampling",
        # beta=2.0,
        answer_mode="raw",
    )

    return sample_set


## ------- Main program -------
if __name__ == "__main__":

    numruns = 10000
    Js = get_Js()

    # bqm = dimod.BQM.from_qubo(Js)
    # sample_set = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=numruns)

    qpu = DWaveSampler(solver={"topology__type": "pegasus"})

    sampler = EmbeddingComposite(qpu)

    print(sampler.properties)

    for k in range(18):
        sample_set = run_on_qpu(Js, hs, sampler)

        print(f"K={k}", sample_set)
        configs = []
        energies = []
        dwave_engs = []

        for i in range(sample_set.record.size):
            for j in range(sample_set.record[i][2]):

                S0 = sample_set._record[i]["sample"]
                dwave_engs.append(sample_set._record[i]["energy"])

                S0d = np.reshape(S0, (Lx, Lx), order="F")
                energy = econf(Lx, J, S0d)

                configs.append(S0)
                energies.append(energy)

        np.save(f"configs_{k}.npy", np.asarray(configs))
        np.save(f"dwave-engs_{k}.npy", np.asarray(dwave_engs))
        np.savetxt(f"energies_{k}.txt", energies)
