# RESOURCE ALLOCATION OPTIMISATION

# Started writing: 11/07/2017

program_name = "FL_GA_RLO_02"

""" IMPORT MODULES
"""

import os
os.system('cls')  # clears screen

import time
import os.path

start_time = time.asctime()
running_time_start = time.clock()

print("Program: FL_GA_RLO")
print("Starts at: " , start_time)
print()
print()Importing modules..."

# DEAP modules to facilitate the genetic algorithm
from deap import algorithms
from deap import base 
from deap import creator # creates the initial individuals
from deap import tools # defines operators

# Modules to handle rasters into arrays
import rasterIO

# Module to handle arrays
import numpy as np

# Module to handle math operators
import math
import random as rndm
from copy import copy

# Module to handle .csv files
import pandas as pd

# Module to handle shape files
import fiona

# Modules for the spatial optimisation framework
# TO BE CREATED:
import Initialise_03 as Init # initialisation module
import FL_Evaluate_02 as Eval # Module to calculate and return fitnesses
import Constraints as Constraint
import FL_Outputs as Output

import gc
gc.disable()

Modules = ['Initialisation', Init.__name__, 'Evaluation', Eval.__name__, 
           'Constraints', Constraint.__name__, 'Output', Output.__name__]
		   
print("All modules imported.")
print()

""" DIRECTORIES
"""
# on P: drive:
Data_Folder     = "../Data/Hull_500m_resolution/"
Code_Folder     = "../Hull_Case_Study/"
Results_Folder  = "./Results_500m_resolution/"
External_Results_Folder = "./Results_500m_resolution/"

""" PROBLEM FORMULATION - General Parameters
"""

# Variables for the search
Spat_Res       = 500		# Defines the spatial resolution (length of the side of a single cell - in meters)
Warehouses_Max = 10			# Maximum number of warehouses
Warehouses_Min = 3			# Minimum number of warehouses
Min_Warh_dist  = 10000.0    # Minimum distance between 2 warehouses - in metres
#################################################################################
# Warehouses_Max should be a function of the budget
#################################################################################

X_quantile = 0.9 # Quantile for travel time opt function. (i.e. how much time XX % of assets are reached from the warehouses)

GEUD_power = 2 # Power of the Generalised Equivalent uniform dose

Problem_Parameters = ['Spatial Resolution (m^2)', Spat_Res, 'Maximum warehouses', Warehouses_Max,
					  'Minimum warehouses', Warehouses_Min, 'GEUD power', GEUD_power]

# Generate availability raster:
if os.path.isfile(os.path.join(Data_Folder, "Available.tif")):
	time_aval = time.asctime()
	print("Skip the generation of Availability Raster because this file already exists in this directory.")
	print()
else:
	time_aval = time.asctime()
	print("Availability raster, starts at: " , time_aval)
	print()
	Init.Generate_Availability(Data_Folder)


#LOOKUP
# To handle the constraints the algorithm uses a lookup for proposed allocation sites.
# The lookup list contains the locations of sites actually available for building warehouses.
# The function called creates a lookup based on our preferences, saves it and returns the list. 

if os.path.isfile(os.path.join(Results_Folder, "lookup.txt")):
	time_lookup_s = time.asctime()
	print("Generation of Lookup starts at: " , time_lookup_s)
	print("Skip the generation of Lookup because this file already exists in this directory.")
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
	time_lookup_e = time.asctime()
	print("Lookup uploaded at: " , time_lookup_e)
else:
	File_centroids = "Available_centroids.shp"
	Lookup = Init.Generate_Lookup(Data_Folder, Results_Folder, File_centroids)
	
# So we know how long to make the chromosome
No_Available = len(Lookup) # number of sites with space for development
print("Number of available cells: " , No_Available )
# (I called "No_Available" what Dan called "No_Undev")


# OUTPUT VARIABLES
# For results 
Sols, Gens = [],[] # Saves all solutions found, saves each generation created                       
# Keep a record of the retaintion after constraints
start = [] # initial array
# Resave the files to contain the arrays
np.savetxt(Results_Folder+'Warehouse_Constraint.txt', start,  delimiter=',', newline='\n')


""" TYPES - creating fitness class, negative weight implies minimisation 
"""

# FITNESS - Defining the number of fitness 
# objectives to minimise or maximise
# Creating types (Fitness, Individual), DEAP documentation: http://deap.readthedocs.io/en/master/tutorials/basic/part1.html
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) # -1.0 for each objective to minimise

# INDIVIDUAL
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMin) #typecode b = integer



""" INITIALISATION - Initially populating the types
"""

toolbox = base.Toolbox()

def Generate_W_Plan(Ind, Warehouses_Max, Warehouses_Min):
	# this function takes as an argument "Ind" which is the creator.individual that takes as an argument the warehouse plan
	# and returns the Individual in which is saved a potential solution
	# Warehouse_Plan is a list of 0s and 1s, where 1=assigned warehouse. To know the coordinates make reference to Lookup list.
	
	# Warehouse_Plan = Init.Generate_WarehousePlan_check_distance(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist)
	
	Warehouse_Plan = Init.Generate_WarehousePlan_Cluster_Ranking(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist)
	# Warehouse_Plan = Init.Generate_WarehousePlan(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder)

	return Ind(Warehouse_Plan) 

toolbox.register("individual", Generate_W_Plan, creator.Individual, Warehouses_Max, Warehouses_Min) # creates an individual = single potential solution
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # creates a population of individuals = set of potential solutions



"""  FUNCTIONS - Evaluate functions and constraint handling
"""

def Evaluate(Warehouse_Plan):
	# Generate the evaluation of functions to minimise/maximise

	Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder) # List of coordinates of proposed sites

	# Dist_Fit       = Eval.Calc_fdist(Results_Folder, Proposed_Sites)   
	# Dist_Fit       = Eval.Calc_fdist_AV_90_dist(Results_Folder, Proposed_Sites, X_quantile, Warehouses_Min, Warehouses_Max)   
	# Dist_Fit       = Eval.Calc_fdist_GEUD(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max, GEUD_power)   
	Dist_Fit = Eval.Calc_fdist_GEUD_checkDist(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max, GEUD_power, Warehouse_Plan, Min_Warh_dist, Lookup)   
	# Dist_Fit       = Eval.Calc_fdist_squared(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max)
	
	# Cost_Fit       = Eval.Calc_fcost(Proposed_Sites, Results_Folder)
	Cost_Fit       = Eval.Calc_fcost_dim(Proposed_Sites, Results_Folder, Data_Folder, Warehouses_Min, Warehouses_Max)
	
	return Dist_Fit, Cost_Fit

Fitnesses = ['fdist', 'fcost']


def Track_Offspring(): # same as Dan
    # Decorator function to save the solutions within the generators
    def decCheckBounds(func):
        def wrapCheckBounds(*args, **kargs):
            offsprings = func(*args, **kargs)
            # Append this generations offspring
            Gens.append(offsprings)
            for child in offsprings:
                # attach each individual solution to solution list. Allows the
				# demonstration of which solutions the Algorithm has investigated.
                Sols.append(child)
            return offsprings
        return wrapCheckBounds
    return decCheckBounds  


""" OPERATORS - Registers Operators and Constraint handlers for the GA
"""

## Evaluator
# Evaluation module - so takes the development plan
toolbox.register("evaluate", Evaluate)

## EVOLUTIONARY OPERATORS - chosen the same of Dan
# Designate the method of crossover
# essentialy takes two points along the array and swaps the warehouses
# Between them. Designating the string name for the output text document
Crossover = "tools.cxTwoPoint"
#toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mate", tools.cxTwoPoint)
# Designate the method of mutation
# Decided to use mutShuffleIndexes which merely moves the 
# elements of the array around
Mutation = "tools.mutShuffleIndexes, indpb=0.1" # indpb - Independent probability for each attribute to be exchanged to another position.
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)

# Selection operator. Either use this or SPEA2 as dealing with multiple OBjs
# See DeB, 2002 for details on NSGA2
Selection = "tools.selNSGA2"
toolbox.register("select", tools.selNSGA2)

Operators = ['Selection', Selection, 'Crossover', Crossover, 'Mutation', Mutation]

# CONSTRAINT HANDLING
# Using a decorator function in order to enforce a constraint of the operation.
# This handles the constraint on the total numBer of warehouses. So the module
# interrupts the selection phase and investigates the solutions selected. If 
# they fail to exceed the minimum warehouse number or exceed the max warehouse number
# its deleted from the gene pool.   
# Moreover to this, each generation is saved to the Gen_list and each generated
# Solution is saved to a sol_list. This for display purposes.

# Constraint to ensure the number of warehouses falls within the targets
# toolbox.decorate("select", Constraint.Check_TotWarehouse_Constraint(Warehouses_Max, Warehouses_Min, Results_Folder))
toolbox.decorate("select", Constraint.Check_Constraint_Select(Warehouses_Max, Warehouses_Min, Results_Folder, Min_Warh_dist), Track_Offspring())

toolbox.decorate("mate", Constraint.Check_Constraint_Mate_Mutate(Warehouses_Max, Warehouses_Min, Results_Folder, Min_Warh_dist))
toolbox.decorate("mutate", Constraint.Check_Constraint_Mate_Mutate(Warehouses_Max, Warehouses_Min, Results_Folder, Min_Warh_dist))

# toolbox.decorate("select", Track_Offspring())


## DAN'S PARAMETERS:
# MU      = 500	# Number of individuals to select for the next generation
# NGEN    = 100   # Number of generations
# LAMBDA  = 500	# Number of children to produce at each generation
# Think this will need to Be really high
# CXPB    = 0.7   # Probability of mating two individuals
# MUTPB   = 0.2   # Probability of mutating an individual

## MY PARAMETERS
"""
MU      = 500	# Number of individuals to select for the next generation
NGEN    = 50   # Number of generations
LAMBDA  = 500	# Number of children to produce at each generation
CXPB    = 0.7   # Probability of mating two individuals
MUTPB   = 0.2   # Probability of mutating an individual
"""
MU      = 1000	# Number of individuals to select for the next generation
NGEN    = 50   # Number of generations
LAMBDA  = 1000  # Number of children to produce at each generation
CXPB    = 0.6   # Probability of mating two individuals
MUTPB   = 0.3   # Probability of mutating an individual

GA_Parameters = ['Generations', NGEN, 'No of individuals to select', MU, 
                 'No of children to produce', LAMBDA, 'Crossover Probability',
                 CXPB, 'Mutation Probability', MUTPB]


def Genetic_Algorithm():    
    # Genetic Algorithm    
    print("Beginning GA operation")
    
    # Create initialised population
    print("Initialising")
    pop = toolbox.population(n=MU)
    
    # hof records a pareto front during the genetic algorithm
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", tools.mean)
    #stats.register("std", tools.std)
    stats.register("min", min)
    #stats.register("max", max)
    
    # Genetic algorithm with inputs
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats= stats, halloffame=hof)
                                                     
    return hof  


if __name__ == "__main__":
	# Returns the saved PO solution stored during the GA
	hof = Genetic_Algorithm()
	
	Complete_Solutions = copy(Sols)
	for PO in hof:
		Complete_Solutions.append(PO)
	
	# Update the results folder to the new directory specifically for this run
	Results_Folder = Output.New_Results_Folder(Results_Folder)    
	
	# Format the solutions so they are compatible with the output functions
	# Gives each a number as well as added the fitness values to from:
	# [ Sol_Num, Sites, Fitnesses]
	frmt_Complete_Solutions = Output.Format_Solutions(Complete_Solutions)
	
	# Extract the minimum and maximum performances for each objective
	# To allow for solutions to be normalised
	MinMax_list = Output.Normalise_MinMax(frmt_Complete_Solutions)
	
	# Normalise the formatted Solution list using the Min and Maxs for 
	# each objective function    
	Normalised_Solutions = Output.Normalise_Solutions(MinMax_list, frmt_Complete_Solutions)
		
	## OLD PLACE OF Output.Output_Run_Details
	
	# Extract all the Pareto fronts using the normalised solutions
	Output.Extract_ParetoFront_and_Plot(Normalised_Solutions, True, External_Results_Folder, Results_Folder, Data_Folder)
	
	# Extract all the Pareto fronts using the solutions retaining their true values.
	Output.Extract_ParetoFront_and_Plot(frmt_Complete_Solutions, False, External_Results_Folder, Results_Folder, Data_Folder)
	
	# Output a file detailing all the run parameters
	running_time_end_s = time.clock() # running time in seconds
	running_time_end_minutes = running_time_end_s/60 # running time in minutes
	run_time = str(int(running_time_end_minutes))
	Output.Output_Run_Details(External_Results_Folder, Results_Folder, Modules, Operators, Problem_Parameters, GA_Parameters, Fitnesses, run_time)

	# Create Shapefiles
	Output.Create_sol_shapefile(Results_Folder, External_Results_Folder)
	
	# Create Distance .csv files for statistics
	Output.Save_dist_W_T(External_Results_Folder, Results_Folder)
	
	# GENERATIONS OUTPUTS
	
	# Create a new array to hold the formatted generations
	frmt_Gens = []    
	for Gen in Gens:
		# For each generation, format it and append it to the frmt_Gens list
		frmt_Gens.append(Output.Format_Solutions(Gen))
	# 
	Output.Extract_Generation_Pareto_Fronts(frmt_Gens,MinMax_list, Results_Folder, Data_Folder, External_Results_Folder)

	end_time = time.asctime()
	
	print("END. end time = ", end_time)
	print("Running time = ", int(running_time_end_minutes), " minutes")