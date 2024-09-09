import pandas as pd
import numpy as np
import time as time
import multiprocessing as mp
from objective_functions import RKFCM_MSCC
from paretoset import paretoset
import copy
import os
import sys

if __name__=="__main__":
    df5 = pd.DataFrame(index=list(range(0,10)),columns=["train J1(clustering compactness)","train J2(misclassification rate)","test J1(clustering compactness)","test J2(misclassification rate)","train Repository","train P_rel","test Repository","test P_rel"])   # df for final results summary
    dataset_start=time.time()
    for set_no in range(0,10):
        start = time.time() # start timer
        for choice in ['train','test']:

            Dataset_Name = sys.argv[1]
            K = int(sys.argv[2]) # no. of cluster centers
            L = int(sys.argv[3]) # no. of classes
            _lambda = float(sys.argv[4])
            
            print("\nRunning for",Dataset_Name,choice,set_no)

            base_path = os.path.dirname(os.path.realpath(__file__))
            X = pd.read_csv(base_path + "/" + Dataset_Name + "/test_sets/train" + str(set_no) + "x.csv",header=None)
            Y = pd.read_csv(base_path + "/" + Dataset_Name + "/test_sets/train" + str(set_no) + "y.csv",header=None)
            X = X.to_numpy()
            Y = Y[0].values

            D = X.shape[1] # no. of dimensions
            P = 500 # no. of particles
            I = 100 # max no. of iterations
            t = 0 # iteration no.

            df1 = pd.DataFrame(index=["x"+str(i) for i in range(1,P+1)], columns=["t = "+str(i) for i in range(1,I+1)]) # df for objective_functions_100_iterations
            df2 = pd.DataFrame(index=["x"+str(i) for i in range(1,P+1)], columns=["t = "+str(i) for i in range(1,I+1)]) # df for P_relation_matrix_100_iterations
            df3 = pd.DataFrame(index=["Repository","Non-Dominated Solutions","P_rel"], columns=["t = "+str(i) for i in range(1,I+1)])   # df for Repository_NonDominatedSolutions_Prel_100_iterations
            df4 = pd.DataFrame(columns=["Repository","Non-Dominated Solutions","P_rel"])   # df for final Repository_NonDominatedSolutions_Prel
            
            print("\nInitializing Positions, Velocities...",end=".. ")

            r1 = 0.5
            r2 = 0.5
            w = 0.4

            POS = np.random.uniform(X.min()-1, X.max()+1, (P,K,D)) # initialize positions
            VEL = np.random.rand(P,K,D) # initialize velocities

            print("Calculating J1, J2...",end=".. ")

            # calculate objective functions J1 and J2
            with mp.Pool() as pool:
                processes = [pool.apply_async(RKFCM_MSCC, args=(X, Y, centers, K, L, _lambda)) for centers in POS]
                result = [p.get() for p in processes]
                pool.close()
                pool.join()

            result = np.array(result,dtype=object)

            CURR_POS = np.array(result[:,0].tolist())   # new updated particle positions

            pbest = np.array(result[:,0].tolist())  # personal best positions = current positions
            pbest_result = np.array(result,dtype=object)
            pbest_result = np.array(pbest_result[:,1:3].tolist())   # personal best objective functions

            print("Extracting Pareto Front...")

            mask = paretoset(result[:,1:3].tolist(), sense=["min", "min"])  # pareto front
            non_dominated_solutions = result[mask][:,1:3]
            Repository = np.array(result[mask][:,0].tolist())
            P_rel_Repo = np.array(result[mask][:,3].tolist())   # P relation matrices of particles in Repository

            print("STARTING MOPSO\n")

            # I number of maximum iterations
            while (t<I):
                
                print("t =",t, end = "   <---------->   ")  # current iteration number

                start_t = time.time()   # start iteration timer
                h = np.random.choice(Repository.shape[0])   # select a particle from Repository randomly
                
                VEL = copy.deepcopy(w*VEL + r1*(pbest - CURR_POS) + r2*(Repository[h] - CURR_POS))  # update velocity
                CURR_POS = copy.deepcopy(CURR_POS + VEL)    # update positions

                # calculate objective functions J1 and J2
                with mp.Pool() as pool:
                    processes = [pool.apply_async(RKFCM_MSCC, args=(X, Y, centers, K, L, _lambda)) for centers in CURR_POS]
                    result = [p.get() for p in processes]
                    pool.close()
                    pool.join()

                result = np.array(result,dtype=object)

                CURR_POS = np.array(result[:,0].tolist())   # new updated particle positions

                mask = paretoset(result[:,1:3].tolist(), sense=["min", "min"])  # pareto front for new positions of particles
                Repository=np.append(Repository,np.array(result[mask][:,0].tolist()),axis=0)    # Insert all the currently nondominated locations into Repository, non_dominated_solutions & P_rel_Repo 
                non_dominated_solutions=np.append(non_dominated_solutions,np.array(result[mask][:,1:3].tolist()),axis=0) 
                P_rel_Repo = np.append(P_rel_Repo,np.array(result[mask][:,3].tolist()),axis=0)

                # Eliminate any dominated locations from the Repository
                mask = paretoset(non_dominated_solutions.tolist(), sense=["min", "min"])
                Repository=Repository[mask]
                non_dominated_solutions=non_dominated_solutions[mask]
                P_rel_Repo = P_rel_Repo[mask]
                
                #  current position of the particle dominates pbest then pbest = CURR_POS
                checks=np.all(pbest_result>=np.array(result[:,1:3].tolist()),axis=1)
                pbest[checks]=copy.deepcopy(CURR_POS[checks])
                pbest_result[checks]=copy.deepcopy(np.array(result[:,1:3].tolist())[checks])

                PATH = base_path+"/"+Dataset_Name+"/"+ choice + "_" + str(set_no)
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                
                # save objective functions J1 and J2    
                df1["t = "+str(t+1)] = result[:,1:3].tolist()
                df1.to_csv(base_path + "/" + Dataset_Name +  "/" + choice + "_" + str(set_no) + "/" + 'objective_functions_100_iterations.csv', index=True, header=True)

                # save P relation matrices
                df2["t = "+str(t+1)] = [str(i) for i in (np.array(result[:,3].tolist()).tolist())]
                df2.to_csv(base_path + "/" + Dataset_Name +  "/" + choice + "_" + str(set_no) + "/" + 'P_relation_matrix_100_iterations.csv', index=True, header=True)

                # save locations, Objective functions J1, J2 & P relation matrices of particles in Repository
                df3.at["Repository","t = "+str(t+1)] = Repository.tolist()
                df3.at["Non-Dominated Solutions","t = "+str(t+1)] = non_dominated_solutions.tolist()
                df3.at["P_rel","t = "+str(t+1)] = P_rel_Repo.tolist()
                df3.to_csv(base_path + "/" + Dataset_Name +  "/" + choice + "_" + str(set_no) + "/" + 'Repository_NonDominatedSolutions_Prel_100_iterations.csv', index=True, header=True)
                
                end_t = time.time()   # end iteration timer
                print(end_t-start_t,"sec")
                print("\n   J1,J2: ",non_dominated_solutions,"\n")
                
                t = t+1

            # save final iteration's Repository, NonDominatedSolutions, Prel matrix
            df4["Repository"] = Repository.tolist()
            df4["Non-Dominated Solutions"] = non_dominated_solutions.tolist()
            df4["P_rel"] = P_rel_Repo.tolist()
            df4.to_csv(base_path + "/" + Dataset_Name +  "/" + choice + "_" + str(set_no) + "/" + 'Final_Repository_NonDominatedSolutions_Prel.csv', index=True, header=True)

            # results summary
            best_accuracy_index = np.argmin(non_dominated_solutions[:,1]) # index with least misclassification rate

            df5.at[set_no, choice + " J1(clustering compactness)"] = non_dominated_solutions[best_accuracy_index][0]
            df5.at[set_no, choice + " J2(misclassification rate)"] = non_dominated_solutions[best_accuracy_index][1]
            df5.at[set_no, choice + " Repository"] = Repository.tolist()[best_accuracy_index]
            df5.at[set_no, choice + " P_rel"] = P_rel_Repo.tolist()[best_accuracy_index]
            # print(df5)
            
            df5.to_csv(base_path + "/" + Dataset_Name +  "/" + 'results_summary.csv', index=True, header=True)

        end = time.time()   # stop timer
        print("Time taken for set",set_no,":",end-start,"sec")    # time taken for each set

    dataset_end=time.time()
    print("Time taken for Dataset",Dataset_Name,":",dataset_end-dataset_start,"sec")    # time taken for whole dataset