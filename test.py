from random import seed
from networkx.readwrite import read_graphml, write_graphml
from ot import unif
import matplotlib.pyplot as plt

from graph import *
from regression import *
from transport import *

from os import mkdir, system
from os.path import exists

## ========== ARGUMENTS FOR THE EXPERIMENT ==========
SEED = 123456789
NMIN = 5
NMAX = 30
NTESTS = 100
NBETAS = 10
NLABELSV = 3
NAME = "TEST"

d = lambda x, y : 0 if x == y else 1

ALPHA = 0.5
NITER = 100

color_map = [
    "#990000",
    "#009900",
    "#000099"
]

seed(SEED)

## ========== INITIALIZING SAVE FOLDERS ==========
if not exists("./save"):
    mkdir("save")

if exists(f"./save/{NAME}"):
    erase = input(f"Experimentation {NAME} already exists. Do you want to erase it ? y/n\n")
    if erase == "y":
        system(f"rm -r ./save/{NAME}")
        mkdir(f"./save/{NAME}")
    else:
        print("Aborted.")
        exit(0)
else:
    mkdir(f"./save/{NAME}")

with open(f"./save/{NAME}/config", "w") as f:
    f.write(
        f"""SEED:{SEED}
NMIN:{NMIN}
NMAX:{NMAX}
NTESTS:{NTESTS}
NBETAS:{NBETAS}
NLABELSV:{NLABELSV}
ALPHA:{ALPHA}
NITER:{NITER}""")

## ========== TESTS ==========

rsquared_mean = 0

for i in range(NTESTS):
    print(f"\n====== {i+1} / {NTESTS} =====\n")
    # ===== Random generation =====
    n1 = randint(NMIN, NMAX)
    n2 = randint(NMIN, NMAX)
    G1 = generate(n1, NLABELSV)
    G2 = generate(n2, NLABELSV)
    mkdir(f"./save/{NAME}/{i+1}")
    mkdir(f"./save/{NAME}/{i+1}/graphs")
    write_graphml(G1, f"save/{NAME}/{i+1}/graphs/G1.graphml")
    write_graphml(G2, f"save/{NAME}/{i+1}/graphs/G2.graphml")

    C1 = all_to_all(G1)
    C2 = all_to_all(G2)
    h1 = unif(G1.number_of_nodes())
    h2 = unif(G2.number_of_nodes())
    M = node_dists(G1, G2, d)

    FGW, _ = fgw(C1, C2, M, h1, h2, ALPHA, NITER)
    with open(f"./save/{NAME}/{i+1}/fgw", "w") as f:
        f.write(str(FGW))

    # ===== Compute FGW for multiple betas =====
    mkdir(f"./save/{NAME}/{i+1}/transform_fgw")
    betas = np.linspace(0, 1, NBETAS)
    Y = []
    he1 = unif(G1.number_of_edges())
    he2 = unif(G2.number_of_edges())

    for j, beta in enumerate(betas):
        print(f"--- beta n° {j+1}/{len(betas)} ---\n")
        mkdir(f"./save/{NAME}/{i+1}/transform_fgw/{j}")
        H1, d2, newh1 = transform(G1, d, lambda x, y : 0, h1, he1, beta)
        H2, d2, newh2 = transform(G2, d, lambda x, y : 0, h2, he2, beta)

        M = node_dists(H1, H2, d2)
        C1_trans = all_to_all(H1)/2
        C2_trans = all_to_all(H2)/2

        FGW_transform, _ = fgw(C1_trans, C2_trans, M, newh1, newh2, ALPHA, NITER)
        with open(f"./save/{NAME}/{i+1}/transform_fgw/{j}/fgw", "w") as f:
            f.write(str(FGW_transform))
        Y.append(FGW_transform/FGW)
    
    # ===== Linear regression =====
    def lin(x, a):
        return (x-1)*a + 1

    a, rsquared = regression(lin, betas, Y)
    rsquared_mean += rsquared

    with open(f"./save/{NAME}/{i+1}/rsquared", "w") as f:
        f.write(str(rsquared))

    # ===== Saving image =====
    plt.figure(i)
    plot(G1, color_map, "G_1", 2, 2, 1)
    plot(G2, color_map, "G_2", 2, 2, 2)
    plt.subplot(2, 2, 3)
    plt.xlabel("beta")
    plt.ylabel("Ratio transformed FGW over FGW")
    plt.plot(betas, Y, 'ro')

    y_reg = [lin(beta, a) for beta in betas]
    plt.plot(betas, y_reg, 'b')
    plt.savefig(f"./save/{NAME}/{i+1}/regression.png")

rsquared_mean /= NTESTS

with open(f"./save/{NAME}/rsquared_mean", "w") as f:
    f.write(str(rsquared_mean))