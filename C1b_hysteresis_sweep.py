# C1b_hysteresis_sweep.py
#------------------------------------------------------------
# This is code to see if there is a hysteresis loop. OP vs K1. Step size of K1 is 0.025.
# Forward sweep, then backward sweep.
# The IC of each subsequent trajectory simulation is the last state of the previous trajectory simulation.
#------------------------------------------------------------
# How to run this code:
# python C1b_hysteresis_sweep.py <ic_omega> <ic_theta> <kk2>
#------------------------------------------------------------
import numpy as np
import networkx as nx
import pandas as pd
from scipy.integrate import odeint
import os
import sys
from datetime import datetime, timedelta
from time import time
#------------------------------------------------------------
tic = time()
#------------------------------------------------------------
now = datetime.now()
print("#----------------------------------------")
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("#----------------------------------------")
#------------------------------------------------------------
# RHS for three node interaction:
def RHS_ThreeNodeInt(theta_List, t,
						N, NodeList, AdjMat, omega_List, K1, DegList,
						jk_dict, K2, k2_i_List):
	IntTerm1 = np.zeros(N, dtype=np.float64) #Pair-wise interactions.
	IntTerm2 = np.zeros(N, dtype=np.float64) #Pair-wise interactions.
	#----------
	# Filling out pair-wise interactions:
	for nn, node in enumerate(NodeList):
		k1_nn = DegList[nn]
		if k1_nn!=0:
			IntTerm1[nn] = K1/k1_nn * np.dot(AdjMat[:,nn], np.sin(theta_List-theta_List[nn]))
	#----------
	theta_Series = pd.Series(theta_List, index=[f"{n}" for n in range(1, N+1)])
	for nn, node in enumerate(NodeList):
		jk_indices = np.array(jk_dict[node])
		j_indices = jk_indices[:,0]
		k_indices = jk_indices[:,1]
		
		IntTerm2[nn] = np.sum(np.sin(theta_List[j_indices.astype(int)-1] + theta_List[k_indices.astype(int)-1] - 2*theta_Series[node]))
	#----------
	IntTerm2 = IntTerm2 * K2/k2_i_List
	#----------
	dydt = omega_List + IntTerm1 + IntTerm2
	return (dydt)
#------------------------------------------------------------
# Dir from where intrinsic freq and theta IC data will be read:
OmegaDir = "DataOmega"
ThetaDir = "DataTheta"
#----------------------------------------
# Make dirs for plots and data:
ThisDir = "C1b_hysteresis" #CHANGE THIS BY HAND!!!

SolDir  = "{}_sol".format(ThisDir)
if not os.path.exists(SolDir): os.makedirs(SolDir)

OrdParamDir  = "{}_r".format(ThisDir)
if not os.path.exists(OrdParamDir): os.makedirs(OrdParamDir)
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

CliqDim = 3 #dimension of cliques of interest.
# Note: 3-dim clique is the same as a 2-dim simplex.
#CHANGE THIS BY HAND!!!
#----------------------------------------
all_cliq_list = np.array(list(nx.enumerate_all_cliques(G))) #all cliques, maximal or not.
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
#------------------------------------------------------------
### Synchronization

# List of coupling constants I want to sweep over.
K1_List = np.linspace(-2.0, 2.0, 161)
K2_List = np.linspace(0.0, 2.0, 21)
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
omega_List = omega_df["omega"].values
#----------------------------------------
#Theta IC:
ic_theta = int(sys.argv[2])
ThetaFileName = "{}/theta_data_ic{}.csv".format(ThetaDir, ic_theta)
theta_df = pd.read_csv(ThetaFileName, index_col=0)
theta_List0 = theta_df["thetaIC"].values
#----------------------------------------
print(f"Omega ic#{ic_omega}")
print(f"Theta ic#{ic_theta}")
print("#----------------------------------------")
#----------------------------------------
# Choose index of 3-point coupling constant:
kk2 = int(sys.argv[3])
K2 = K2_List[kk2]
print(f"K2 = {K2}")
print("#----------------------------------------")
#----------------------------------------
# Calculate order parameter:
print("K1\tK2\tTimeAvg\t\tStdDev\t\tStdErr\t\tWidth")
#----------------------------------------
# Forward sweep:
OutFileName_forward = f"{OrdParamDir}/{ThisDir}_forward_r_data_omega{ic_omega}_theta{ic_theta}_K2_{kk2}.csv"
OutFile_forward = open(OutFileName_forward, "w")
OutFile_forward.write("K1,K2,TimeAvg,StdDev,StdErr,Width\n")

for kk1, K1 in enumerate(K1_List):
	# Run in-built scipy ODE solver to get solution:
	sol = odeint(RHS_ThreeNodeInt, theta_List0, t_List,
					args=(N, NodeList, A, omega_List, K1, DegList,jk_dict, K2, k2_i_List))
	#----------------------------------------
	# Put solution in pandas df:
	sol_dict = {}
	for osc_index, node in enumerate(NodeList):
		sol_dict["osc"+node] = sol[:,osc_index]
	sol_df =  pd.DataFrame.from_dict(sol_dict, dtype=np.float64)
	sol_df = sol_df%(2*np.pi)

	# Print sol to file:
	sol_FileName_npy = f"{SolDir}/{ThisDir}_forward_sol_omega{ic_omega}_theta{ic_theta}_K1_{kk1}_K2_{kk2}.npy"
	sol_ToSave = np.array([sol_df.values[0], sol_df.values[-1]])
	np.save(sol_FileName_npy, sol_ToSave)
	# np.save(sol_FileName_npy, sol_df.values)
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

	OutFile_forward.write("{},{},{},{},{},{}\n".format(K1, K2, r_TimeAvg, r_std, r_std_error, r_MaxMin))
	print("{}\t{}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}".format(K1, K2, r_TimeAvg, r_std, r_std_error, r_MaxMin))
	#----------------------------------------
	# Use last state of the system as IC for next calculation:
	theta_List0 = sol_df.values[-1]
	#----------------------------------------
OutFile_forward.close()
#----------------------------------------
print("#----------------------------------------")
#----------------------------------------
# Backward sweep:
OutFileName_backward = f"{OrdParamDir}/{ThisDir}_backward_r_data_omega{ic_omega}_theta{ic_theta}_K2_{kk2}.csv"
OutFile_backward = open(OutFileName_backward, "w")
OutFile_backward.write("K1,K2,TimeAvg,StdDev,StdErr,Width\n")

for kk1, K1 in enumerate(sorted(K1_List, reverse=True)):
	# Run in-built scipy ODE solver to get solution:
	sol = odeint(RHS_ThreeNodeInt, theta_List0, t_List,
					args=(N, NodeList, A, omega_List, K1, DegList,jk_dict, K2, k2_i_List))
	#----------------------------------------
	# Put solution in pandas df:
	sol_dict = {}
	for osc_index, node in enumerate(NodeList):
		sol_dict["osc"+node] = sol[:,osc_index]
	sol_df =  pd.DataFrame.from_dict(sol_dict, dtype=np.float64)
	sol_df = sol_df%(2*np.pi)

	# Print sol to file:
	sol_FileName_npy = f"{SolDir}/{ThisDir}_backward_sol_omega{ic_omega}_theta{ic_theta}_K1_{kk1}_K2_{kk2}.npy"
	sol_ToSave = np.array([sol_df.values[0], sol_df.values[-1]])
	np.save(sol_FileName_npy, sol_ToSave)
	# np.save(sol_FileName_npy, sol_df.values)
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

	OutFile_backward.write("{},{},{},{},{},{}\n".format(K1, K2, r_TimeAvg, r_std, r_std_error, r_MaxMin))
	print("{}\t{}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}".format(K1, K2, r_TimeAvg, r_std, r_std_error, r_MaxMin))
 	#----------------------------------------
 	# Use last state of the system as IC for next calculation:
	theta_List0 = sol_df.values[-1]
	#----------------------------------------
OutFile_backward.close()
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
