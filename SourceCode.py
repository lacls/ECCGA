import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
import random
import pandas as pd
from copy import deepcopy
import math
import argparse
parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("-d", type=int)
parser.add_argument("-i", type=int)

args = parser.parse_args()
i=args.i
d = [args.d]
Bits_array=[]
#BLOCK1 - Population, fitness Score
def inital_pop(length, number,seed):#number is odd
    pop=[]
    if number%2!=0:
        while len(pop)!=number-1:
            pop.append(np.zeros(length))
            pop.append(np.ones(length))
        pop.append(np.zeros(length))
    else:
        while len(pop)!=number:
            pop.append(np.zeros(length))
            pop.append(np.ones(length))
    return randomized(np.array(pop),seed)
        
def randomized(pop,seed):
    np.random.seed(seed)
    for x in range(len(pop[0])):
        #np.random.seed(seed)
        pop[:,x]= np.random.permutation(pop[:,x])
#        https://stackoverflow.com/questions/47742622/np-random-permutation-with-seed
    return pop

        
def fitness_calculation(pop,method):
    fitness_value=[]
    if method=='OneMax':
        for j in range(len(pop)):
            count=0
            for k in pop[j]:
                if k==1:
                    count+=1
            fitness_value.append(count)
    else:
           fitness_value_for_each_Trap=[]
           for num_par in range(len(pop)):
               for j in range(0,len(pop[num_par]),5):
                   count_one=list(pop[num_par][j:j+5]).count(1)
                   if count_one<5:
                        fitness_value_for_each_Trap.append(4-count_one)
                   else:
                        fitness_value_for_each_Trap.append(5)
               fitness_value.append(sum(fitness_value_for_each_Trap))
               fitness_value_for_each_Trap.clear()
    return fitness_value

def first_generation(pop,fitness_Score):#sort_Valued
    sorted_fitness = sorted([[pop[x], fitness_Score[x]]
        for x in range(len(pop))], key=lambda x: x[1])
    population = [sorted_fitness[x][0] 
        for x in range(len(sorted_fitness))]
    fitness_Score = [sorted_fitness[x][1] 
        for x in range(len(sorted_fitness))]
    return {'Individuals': population, 'Fitness': sorted(fitness_Score)}

#def mating(parents_,method):
#    offsprings=[]
#    for x in range(0,len(parents_),2):
#        if method == 'Single Point':
#            pivot_point = random.randint(1,len(parents_[0])-1)
#            offsprings.append(np.array(list(parents_[x][0:pivot_point])+list(parents_[x+1][pivot_point:])))
#            offsprings.append(np.array(list(parents_[x+1][0:pivot_point])+list(parents_[x][pivot_point:])))
#        elif method=='UX':
#            child1= np.array(parents_[x])
#            child2= np.array(parents_[x+1])
#            for j in range(len(parents_[0])):
#                coin=random.randint(0,1)
#                if coin==1:
#                    child1[j],child2[j]=child2[j].copy(),child1[j].copy()
#            offsprings.append(child1)
#            offsprings.append(child2)
#    return offsprings
    
'''
ECGA
'''
def GetTheArray(arr, n):  
    arr_=[]
    for i in range(0, n):  
        arr_.append(arr[i])  
    Bits_array.append(arr_)
  
# Function to generate all binary strings  
def generateAllBinaryStrings(n, arr, i):  
  
    if i == n: 
        GetTheArray(arr, n)  
        return
      
    # First assign "0" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)  
  
    # And then assign "1" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)  
def listofNbits(n):
    global Bits_array
    Bits_array.clear()
    arr = [None] * n  
    generateAllBinaryStrings(n, arr, 0)
    
def MPM(population):
    import math
    population_=np.asarray(population)
    strt=list(range(0,len(population[0])))
    MDL_BestOne=10000000000000000
    success=False
    Sum_MC=0
    offical_model=[]
    subdict=[]
    while success is False:
        for i in range(len(strt)-1):
            k=i+1
            if type(strt[i]) is not int:
                value_1=list(strt[i])
            else:
                value_1=list([strt[i]])
        
            while k !=len(strt):
                arr=[]
                if type(strt[k]) is not int:
                    value_2=list(strt[k])
                else:
                    value_2=list([strt[k]])
                arr.append(value_1+value_2)
                Sum_MC+=(pow(2,len(arr[0]))-1)
                for index in range(len(strt)):
                    if index not in list([k,i]):
                        arr.append(strt[index])
                        if type(strt[index]) is int:
                            Sum_MC+=(pow(2,1)-1)
                        else:
                            Sum_MC+=(pow(2,len(strt[index]))-1)
                MC_model= (math.log2(len(population_[:,1])+1))*Sum_MC
                CPC_model,roulete_selection=CPC_Calculation(arr,population_,len(population_[:,1]))
                if MDL_BestOne > MC_model+CPC_model:
                    MDL_BestOne=MC_model+CPC_model
                    offical_model=arr.copy()
                    subdict=roulete_selection.copy()
                    success=True
                k+=1
                Sum_MC=0
        if success==True:
            # print(offical_model)
            # print(MDL_BestOne)
            strt=offical_model.copy()
            success=False
            continue
        else:
            return offical_model,subdict
def CPC_Calculation(arr,population, num_individuals):
    Sum_CPC=0
    roulete_selection=[]
    bits_Array=[]
    subdict=[]
    for value in arr:
        if type(value) is not int: 
            listofNbits(len(value))
            compare_Values=np.array([population[:,index] for index in value ])
            for value_bits in Bits_array:
                count_frequency=0
                for m in range(len(compare_Values[0])):
                    if False in (value_bits == compare_Values[:,m]):
                        continue
                    else:
                        count_frequency+=1
                Px=count_frequency/num_individuals
                roulete_selection.append(Px)
                bits_Array.append(value_bits)
                if Px!=0:
                    Sum_CPC+=Px*math.log2(1/Px)
            # idx   = np.argsort(roulete_selection)
            # roulete_selection=list(np.array(roulete_selection)[idx])
            # bits_Array=list(np.array(bits_Array)[idx])
            subdict.append({'Bits array':bits_Array.copy(),'Value':roulete_selection.copy()})
            bits_Array.clear()
            roulete_selection.clear()
            
        else:
            listofNbits(1)
            for value_bits in Bits_array:
                value_calculated=(list(population[:,value]).count(value_bits))/num_individuals
                roulete_selection.append(value_calculated)
                bits_Array.append(value_bits)
                Sum_CPC+=value_calculated
            # idx   = np.argsort(roulete_selection)
            # roulete_selection=list(np.array(roulete_selection)[idx])
            # bits_Array=list(np.array(bits_Array)[idx])
            subdict.append({'Bits array':bits_Array.copy(),'Value':roulete_selection.copy()})
            bits_Array.clear()
            roulete_selection.clear()                        
    return num_individuals*Sum_CPC,subdict

def matingECGA(population,offical_model,subdict):
    storage=deepcopy(population)
    index_for_subdict=-1
    for x in offical_model:
        index_for_subdict+=1
        for rows in range(len(storage)):
            r=random.uniform(0, 1)
            compared_Value=subdict[index_for_subdict]['Value'][0]
            for index in range(len(subdict[index_for_subdict]['Bits array'])):
                if r>compared_Value:
                    compared_Value+=subdict[index_for_subdict]['Value'][index+1]
                    continue
                else:
                    h=0
                    if type(x) is not int:
                        for t in x:
                            storage[rows][t]=subdict[index_for_subdict]['Bits array'][index][h]
                            h+=1
                    else:
                            storage[rows][t]=subdict[index_for_subdict]['Bits array'][index][h]
                    break
    return list(storage)
def all_same(items):
    return all(x == items[0] for x in items)



def calculate(num_d,N_upper,seed,eval_method):
    count=1 #complete one round
    fitness_proposed=[d[num_d]]*N_upper
    population=inital_pop(d[num_d],N_upper,seed)            
    fitness_Score=fitness_calculation(population,eval_method)
    #gen=first_generation(population,fitness_Score)
    parents=list(population)
    #    print('Method: ',method)
    #    print('Evaluation Method: ',eval_method)
    #    print('Population size: ',population.shape)
    #    print('--------------------------------')
    while fitness_Score != fitness_proposed:
        if all_same(fitness_Score)== True:
            return {'Eval_is_Called':N_upper+count*(2*N_upper),'Status':0,'Shape': population.shape,'Seed':seed}
        offical_model,subdict=MPM(deepcopy(parents))
        offsprings=matingECGA(parents,offical_model,subdict)
        parents_off= parents+offsprings
        fitness_Score_POPO=fitness_calculation(parents_off,eval_method)
        selected_individuals=[]
        selected_individuals_fitness=[]

        while len(selected_individuals)!=len(parents):
            for j in range(0,len(parents_off),4):            
                gen_POPO=first_generation(parents_off[j:j+4],fitness_Score_POPO[j:j+4])
                selected_individuals.append(gen_POPO['Individuals'][3])
                selected_individuals_fitness.append(gen_POPO['Fitness'][3])
            if len(selected_individuals)!=len(parents):
                fitness_gen= list(zip(parents_off, fitness_Score_POPO))
                random.shuffle(fitness_gen)
                parents_off, fitness_Score_POPO = zip(*fitness_gen)
            else:
                break
        #gen=first_generation(selected_individuals,selected_individuals_fitness)
        fitness_Score=list(selected_individuals_fitness)
        parents=selected_individuals
        count+=1
        if fitness_Score==fitness_proposed:
            return {'Eval_is_Called':N_upper+count*(2*N_upper),'Status':1,'Shape': population.shape,'Seed':seed}

def run_func(eval_method):
    MSSV=17520669
    results_single_point=[]
    N_upper_single=[]
    N_upper_single_avg=[]
    for x in range(len(d)):
        for bi_section in range(0,i):
            '''Step 1: Find upper bound of MRPS'''
            
            Success=False
            N_upper=4
            while Success==False:
                for t in range(0,10):
                    random.seed(MSSV+t)
                    results_calculated=calculate(x,N_upper,MSSV+t,eval_method)
                    if results_calculated['Status']==0:
                        N_upper=N_upper*2
                        if N_upper >16000:
                            Success=True
                            break
                        results_single_point.clear()
                        break
                    else:
                        results_single_point.append(results_calculated)   
                if len(results_single_point)==10:
                    Success=True   
            '''Step 2: Find value of MRPS'''
            # print('***********STEP 2***********')
            if N_upper > 16000:
              break
            N_lower=N_upper/2
            while ((N_upper - N_lower)/N_upper)>0.1:
                N= (N_upper+N_lower)/2
                Success=False
                results_single_point.clear()
                for t in range(0,10):
                    random.seed(MSSV+t)
                    results_calculated=calculate(x,int(N),MSSV+t,eval_method)
                    if results_calculated['Status']==0:
                        results_single_point.clear()
                        break
                    else:
                        results_single_point.append(results_calculated)
                if len(results_single_point)==10:
                    Success=True
                    
                if Success==True:
                    N_upper=N
                else:
                    N_lower=N
                    
                if (N_upper-N_lower)<=2:
                    break
            # print('NEW ROUND')
            N_upper_single.append(int(N_upper))
            MSSV+=10
        if N_upper > 16000:
          N_upper_single_avg.append(np.zeros(10))
          continue
        N_upper_single_avg.append(N_upper_single.copy())
        N_upper_single.clear()
        results_single_point.clear()
        MSSV=17520669
    return N_upper_single_avg
def main():
    
    return run_func('OneMax')
    
#    N_upper_UX_OneMax=run_func('UX','OneMax')
#    N_upper_UX_OneMax=pd.DataFrame(N_upper_UX_OneMax)
#    N_upper_UX_OneMax.to_csv('/content/drive/My Drive/Storage_Collab/N_upper_UX_OneMax.csv') 
#    
#    N_upper_single_TrapFunc =run_func('Single Point','Trap')
#    N_upper_single_TrapFunc=pd.DataFrame(N_upper_single_TrapFunc)
#    N_upper_single_TrapFunc.to_csv('/content/drive/My Drive/Storage_Collab/N_upper_single_TrapFunc.csv') 
#
#    N_upper_UX_TrapFunc=run_func('UX','Trap')
#    N_upper_UX_TrapFunc=pd.DataFrame(N_upper_UX_TrapFunc)
#    N_upper_UX_TrapFunc.to_csv('/content/drive/My Drive/Storage_Collab/N_upper_UX_TrapFunc.csv') 
#    print('Complete OneMax')
N_upper_single_OneMax= main()
print(N_upper_single_OneMax)
#if __name__ == "__main__":
#  #main()

#N_upper_single_TrapFunc=pd.read_csv("output_reference/N_upper_single_TrapFunc_Offical.csv")
#
#
#eval_called_group=[]
#eval_called_final=[]
#for k in range(5):
#  for index_columns in range(10):
#    eval_Called=[]
#    for t in range(0,10):
#        random.seed(MSSV+t)
#        results_calculated=calculate("Single Point",k,N_upper_single_TrapFunc[str(index_columns)][k],MSSV+t,"OneMax")
#        eval_Called.append(results_calculated['Eval_is_Called'])
#    eval_called_group.append(np.sum(eval_Called))
#    eval_Called.clear()
#    MSSV+=10
#  eval_called_final.append(eval_called_group.copy())
#  eval_called_group.clear()
#  MSSV=17520669
#eval_called_final=pd.DataFrame(eval_called_final)
#eval_called_final.to_csv('/content/drive/My Drive/Storage_Collab/EA/BT3/eval_called_N_upper_single_OneMax.csv',index=False)
#population=inital_pop(4,4,17520669) 
#parents=list(population)
#a = np.array([[1,2,6],[4,5,8],[8,3,5],[6,5,4]])
#print(a[:,0])
#
#population[:,2]
