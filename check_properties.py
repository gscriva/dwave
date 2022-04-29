import numpy as np

from dwave.system import DWaveSampler, LeapHybridSampler, LeapHybridCQMSampler


def get_Js(J, Lx):
    Js = {}
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
    np.savetxt(f"{Lx**2}spins-uniform-1nn.txt", txtarr, fmt=['%d', '%d', '%1.10f'])
    return Js

if __name__ == "__main__":
    qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})

    print(qpu_advantage.properties["default_programming_thermalization"])
    print(qpu_advantage.properties["default_readout_thermalization"])
    print(qpu_advantage.properties["readout_thermalization_range"])


    Lx = 3
    N = Lx ** 2
    np.random.seed(12345)
    J = (np.random.uniform(-1., 1., size=(N - Lx, 2))) * - 1.0

    _ = get_Js(J, Lx)    

