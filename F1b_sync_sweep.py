# F1b_sync_sweep.py
#------------------------------------------------------------
# Synchronization simulation with 4-node (or 3-simplex, or 4-clique) interactions.
# Intrinsic frequency of all the oscillators is taken to be identical.
# This is code sweeps across K1 and calculates OP.
# This is not a hysteresis sweep, ie IC is not last state of previous simulation.
# K1_List = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.]
#------------------------------------------------------------
# How to run this code:
# python F1b_sync_sweep.py <ic_omega> <ic_theta> <kk2> <kk3>
#------------------------------------------------------------
import numpy as np
import networkx as nx
import pandas as pd
from scipy.integrate import odeint
import os
import sys
from datetime import datetime, timedelta
from time import time
import matplotlib.pyplot as plt
#------------------------------------------------------------
tic = time()
#------------------------------------------------------------
now = datetime.now()
print("#----------------------------------------")
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("#----------------------------------------")
#------------------------------------------------------------
# RHS for four-node interaction:
def RHS_FourNodeInt(theta_List, t,
						N, NodeList, AdjMat, omega_List, K1, DegList,
						jk_dict, K2, k2_i_List,
						jkl_dict, K3, k3_i_List):
	IntTerm1 = np.zeros(N, dtype=np.float64) #Pair-wise interactions.
	IntTerm2 = np.zeros(N, dtype=np.float64) #2-simplex interactions.
	IntTerm3 = np.zeros(N, dtype=np.float64) #3-simplex interactions.
	#----------
	# Pair-wise interactions:
	for nn, node in enumerate(NodeList):
		k1_nn = DegList[nn]
		if k1_nn!=0:
			IntTerm1[nn] = K1/k1_nn * np.dot(AdjMat[:,nn], np.sin(theta_List-theta_List[nn]))
	#----------
	# 2-simplex interactions:
	theta_Series = pd.Series(theta_List, index=[f"{n}" for n in range(1, N+1)])
	for nn, node in enumerate(NodeList):
		jk_indices = np.array(jk_dict[node])
		j_indices = jk_indices[:,0]
		k_indices = jk_indices[:,1]
		
		IntTerm2[nn] = np.sum(np.sin(theta_List[j_indices.astype(int)-1] + theta_List[k_indices.astype(int)-1] - 2*theta_Series[node]))
	#---
	IntTerm2 = IntTerm2 * K2/k2_i_List
	#----------
	# 3-simplex interactions:
	theta_Series = pd.Series(theta_List, index=[f"{n}" for n in range(1, N+1)])
	for nn, node in enumerate(NodeList):
		jkl_indices = np.array(jkl_dict[node])
		j_indices = jkl_indices[:,0]
		k_indices = jkl_indices[:,1]
		l_indices = jkl_indices[:,2]
		
		IntTerm3[nn] = np.sum(np.sin(theta_List[j_indices.astype(int)-1] + theta_List[k_indices.astype(int)-1] + theta_List[l_indices.astype(int)-1] - 3*theta_Series[node]))
	#---
	IntTerm3 = IntTerm3 * K3/k3_i_List
	#----------
	dydt = omega_List + IntTerm1 + IntTerm2 + IntTerm3
	return (dydt)
#------------------------------------------------------------
# Dir from where intrinsic freq and theta IC data will be read:
OmegaDir = "DataOmega"
ThetaDir = "DataTheta"
#----------------------------------------
# Make dirs for plots and data:
ThisDir = "F1b_FourNodeIntSync" #CHANGE THIS BY HAND!!!

SolDir  = "{}_sol".format(ThisDir)
if not os.path.exists(SolDir): os.makedirs(SolDir)

OrdParamDir  = "{}_r".format(ThisDir)
if not os.path.exists(OrdParamDir): os.makedirs(OrdParamDir)

PlotDir  = "{}_Plots".format(ThisDir)
if not os.path.exists(PlotDir): os.makedirs(PlotDir)
#------------------------------------------------------------
### The network

MyEdgeList_FileName = "dclnet5_ni5_p07.edgelist"
#CHANGE THIS BY HAND!!!
#----------------------------------------
G = nx.read_weighted_edgelist(MyEdgeList_FileName, nodetype=str, create_using=nx.Graph)
#----------------------------------------
N = G.number_of_nodes() #Number of nodes in the network.
M = G.number_of_edges() #Number of edges in the network.
print("N={}, M={}".format(N, M))
#----------------------------------------
NodeList = ['{}'.format(_) for _ in range(1, N+1)]
#I can do this, because I know that the nodes are numbered 1, 2, ..., N, and are of type string.
#----------------------------------------
DegList = [G.degree(node) for node in NodeList]
#----------------------------------------
# Adjacency matrix of network:
A = nx.to_numpy_array(G, nodelist=NodeList, weight=None)
#------------------------------------------------------------
### The corresponding simplicial structure

all_cliq_list = np.array(list(nx.enumerate_all_cliques(G)), dtype=object) #all cliques, maximal or not.
#----------------------------------------
## 2-simplex structure:

CliqDim = 3 #dimension of cliques of interest.
# Note: 3-dim clique is the same as a 2-dim simplex.
#CHANGE THIS BY HAND!!!
#----------------------------------------
# Cliques of dimension=CliqDim
cliq3_list = all_cliq_list[np.array(list(map(len, all_cliq_list)))==CliqDim]
#----------------------------------------
print(f"Number of cliques of dim-{CliqDim} (ie simplices of dim-{CliqDim-1})={len(cliq3_list)}")
#----------------------------------------
# Calculate the avg 2-simplex degree across the full network:
cliq3_GenDeg_dict = {}
for node in NodeList:
	cliq3_deg = sum([node in cliq for cliq in cliq3_list]) #number of 3-cliques that this node participates in.
	cliq3_GenDeg_dict[node] = cliq3_deg

k2_i_List = np.array(list(cliq3_GenDeg_dict.values()))
k2_i_avg = k2_i_List[k2_i_List.nonzero()].mean()

print(f"Avg {CliqDim}-clique degree = {k2_i_avg}")
print("#----------------------------------------")
#----------------------------------------
# Dict for j, k indices of B_ijk:
jk_dict = {}
for node in NodeList:
	jk_list_for_this_node = []
	for cliq3 in cliq3_list:
		if node in cliq3:
			jk_list_for_this_node.append( list(set(cliq3)-set([node])) )
	jk_dict[node] = jk_list_for_this_node
#----------------------------------------
## 3-simplex structure:

CliqDim = 4 #dimension of cliques of interest.
# Note: 4-dim clique is the same as a 3-dim simplex.
#CHANGE THIS BY HAND!!!
#----------------------------------------
# Cliques of dimension=CliqDim
cliq4_list = all_cliq_list[np.array(list(map(len, all_cliq_list)))==CliqDim]
#----------------------------------------
print(f"Number of cliques of dim-{CliqDim} (ie simplices of dim-{CliqDim-1})={len(cliq4_list)}")
#----------------------------------------
# Calculate the avg 3-simplex degree across the full network:
cliq4_GenDeg_dict = {}
for node in NodeList:
	cliq4_deg = sum([node in cliq for cliq in cliq4_list]) #number of 4-cliques that this node participates in.
	cliq4_GenDeg_dict[node] = cliq4_deg

k3_i_List = np.array(list(cliq4_GenDeg_dict.values()))
k3_i_avg = k3_i_List[k3_i_List.nonzero()].mean()

print(f"Avg {CliqDim}-clique degree = {k3_i_avg}")
print("#----------------------------------------")
#----------------------------------------
# Dict for j, k, l indices of C_ijkl:
jkl_dict = {}
for node in NodeList:
	jkl_list_for_this_node = []
	for cliq4 in cliq4_list:
		if node in cliq4:
			jkl_list_for_this_node.append( list(set(cliq4)-set([node])) )
	jkl_dict[node] = jkl_list_for_this_node
#------------------------------------------------------------
### Synchronization

# List of coupling constants I want to sweep over.
K1_List = np.linspace(-1., 0., 11) #CHANGE THIS BY HAND!!!
K2_List = np.linspace(0.0, 2.0, 21)
K3_List = np.linspace(0.0, 2.0, 21)
#----------------------------------------
# Time:
t_start = 0
t_stop = 500
dt = 0.01

t_List = np.arange(t_start, t_stop, dt)

N_t = len(t_List)
#----------------------------------------
# Length of end part of order parameter TS on which time avg will be calculated:
N_r = int((2./5)*N_t)
print("N_t = {}".format(N_t))
print("N_r = {}".format(N_r))
print("#----------------------------------------")
# print("\n")
#----------------------------------------
# Instrinsic freq:
ic_omega = int(sys.argv[1])
OmegaFileName = "{}/omega_data_ic{}.csv".format(OmegaDir, ic_omega)
omega_df = pd.read_csv(OmegaFileName, index_col=0)
omega_List = np.array( [omega_df["omega"].values[0]] *N)
#----------------------------------------
#Theta IC:
ic_theta = int(sys.argv[2])
ThetaFileName = "{}/theta_data_ic{}.csv".format(ThetaDir, ic_theta)
theta_df = pd.read_csv(ThetaFileName, index_col=0)
theta_List0 = theta_df["thetaIC"].values
#----------------------------------------
print(f"Omega ic#{ic_omega}, omega={omega_List[0]}")
print(f"Theta ic#{ic_theta}")
print("#----------------------------------------")
#----------------------------------------
# Choose index of 2-simplex and 3-simplex coupling constants:
kk2 = int(sys.argv[3])
kk3 = int(sys.argv[4])

K2 = K2_List[kk2]
K3 = K3_List[kk3]

print(f"K2 = {K2}")
print(f"K3 = {K3}")
print("#----------------------------------------")
#----------------------------------------
# Calculate order parameter:
print("K1\tK2\tK3\tTimeAvg\t\tStdDev\t\tStdErr\t\tWidth")
#----------------------------------------
# Sweep:
OutFileName = f"{OrdParamDir}/{ThisDir}_r_data_omega{ic_omega}_theta{ic_theta}_K2_{kk2}_K3_{kk3}.csv"
OutFile = open(OutFileName, "w")
OutFile.write("K1,K2,K3,TimeAvg,StdDev,StdErr,Width\n")

for kk1, K1 in enumerate(K1_List):
	# Run in-built scipy ODE solver to get solution:
	sol = odeint(RHS_FourNodeInt, theta_List0, t_List,
					args=(N, NodeList, A, omega_List, K1, DegList, jk_dict, K2, k2_i_List, jkl_dict, K3, k3_i_List))
	#----------------------------------------
	# Put solution in pandas df:
	sol_dict = {}
	for osc_index, node in enumerate(NodeList):
		sol_dict["osc"+node] = sol[:,osc_index]
	sol_df =  pd.DataFrame.from_dict(sol_dict, dtype=np.float64)
	sol_df = sol_df%(2*np.pi)

	# Print sol to file:
	sol_FileName_npy = f"{SolDir}/{ThisDir}_sol_omega{ic_omega}_theta{ic_theta}_K1_{kk1}_K2_{kk2}_K3_{kk3}.npy"
	sol_ToSave = np.array([sol_df.values[0], sol_df.values[-1]])
	np.save(sol_FileName_npy, sol_ToSave)
	#----------------------------------------
	# Calculate order parameter:
	x_pos_df = np.cos(sol_df)
	y_pos_df = np.sin(sol_df)
	#---
	r_x_df = x_pos_df.sum(axis=1)/N
	r_y_df = y_pos_df.sum(axis=1)/N

	r_df = np.sqrt(r_x_df**2 + r_y_df**2)
	#----------------------------------------
	r_ForCalc = r_df.values[-N_r:]

	r_TimeAvg = np.mean(r_ForCalc)
	r_std = np.std(r_ForCalc, ddof=1)
	r_std_error = r_std/np.sqrt(N_r)
	r_MaxMin = max(r_ForCalc) - min(r_ForCalc)

	OutFile.write("{},{},{},{},{},{},{}\n".format(K1, K2, K3, r_TimeAvg, r_std, r_std_error, r_MaxMin))
	print("{}\t{}\t{}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}".format(K1, K2, K3, r_TimeAvg, r_std, r_std_error, r_MaxMin))
	#----------------------------------------
	# Plot OP vs time:
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 2, 1])

	ax.plot(t_List, r_df, "-")
	ax.plot(t_List[-N_r:], r_df[-N_r:], "r-")

	ax.set_xlim([t_List[0], t_List[N_t-1]])
	ax.set_ylim([-0.1, 1.1])
	ax.grid(True)

	ax.set_ylabel("Order Parameter, r", fontsize=15)
	ax.set_xlabel("time", fontsize=15)
	ax.set_title(rf"$K_1$={K1}, $K_2$={K2}, $K_3$={K3}, #timesteps={N_t}")

	FigName = f"{PlotDir}/{ThisDir}_OrderParamVsTime_omega{ic_omega}_theta{ic_theta}_K1_{kk1}_K2_{kk2}_K3_{kk3}.png"
	plt.savefig(FigName, format="png", transparent=False, bbox_inches="tight", pad_inches=0.1)
	#----------------------------------------
OutFile.close()
#------------------------------------------------------------
print("#----------------------------------------")
toc = time()
TimeTaken_in_sec = toc - tic
# print(f"Time taken: {toc-tic:.2f} s")
print(f"Time taken:")
print(timedelta(seconds=TimeTaken_in_sec))
#------------------------------------------------------------
print("#----------------------------------------")
print("Done!")
print("#----------------------------------------")
