# -*- coding: utf-8 -*-
"""
Objectives currently optimised include:
    1. Travel times
    2. Costs = travel costs, land costs

"""

"""
# sys.path.append:
# works as it will search the anaconda directory (which is the default python installation)
# for any libraries you try to import, and when it doesn’t find them, it will then search the python27 directory
# as we have added this as a path to look in. The only downside to this is that is a version of a library is in
# the anaconda directory, it will use that, rather than the version you have coped over.
import sys
sys.path.append('C:/Python27/Lib/site-packages')
"""

# program_name = "Evaluate_02"

# import os
# os.system('cls')  # clears screen

import time
start_time = time.asctime()

# print "Program: " , program_name
# print "Starts at: " , start_time
print
# print "Importing modules..."
import fiona
import pandas as pd
import copy
import sys
import Initialise_04 as Init
import Constraints
import math
import numpy as np
# print "Modules imported."
print

# Data_Folder     = "P:/RLO/Python_Codes/Data/Hull/"
# Results_Folder  = "P:/RLO/Python_Codes/Hull_Case_Study/Results/"
# File_centroids  = "Available_centroids.shp"
# Lookup = Init.Generate_Lookup(Data_Folder, Results_Folder, File_centroids)
# Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
# No_Available = len(Lookup)
# Warehouses_Max = 10
# Warehouses_Min = 2
# Warehouse_Plan = Init.Generate_WarehousePlan_check_distance(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist)
# Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder) # List of coordinates of proposed sites

def Calc_fdist_maxTT(Results_Folder, Proposed_Sites):
	# Function that calculates fdist as the MAX TRavTime (in minutes) from a SINGLE proposed site to the farthest target
	
	dist_dict_file = 'Dictionary_cells_targets'
	available_centroids_file = 'Available_centroids.shp'
	
	# Create the dataframe from csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets' coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)

	max_dist = Selected_points_trgt_df['Tot_dist'].max() # maximum value of the column

	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	########################################################################################
	if len(Proposed_Sites) == 0:
		fdist = 100000 # Meaningless very high number
	else:
		fdist = (max_dist/60.0) # Max TRavTime (in minutes) from a SINGLE proposed site to the farthest assigned target

	return fdist


def Calc_fdist_AV_90_dist(Results_Folder, Proposed_Sites, X_quantile, Min_W, Max_W):
	# Function that calculates fdist as average distance of the XXth percentile of assets
	
	dist_dict_file = 'Dictionary_cells_targets'
	available_centroids_file = 'Available_centroids.shp'
	
	# Create the dataframe from csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	# sort data frame on targets' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) 

	# Keeps the first row of each target (which contains the closest point)
	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True)

	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # WRONG
	Selected_points_trgt_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	""" WRONG	
	# Create a Dataframe with only the coordinates of strat infr assets and Tot_dist
	A_W_df = Selected_points_trgt_df[['X_Target','Y_Target','Tot_dist']]
	
	# Calculate the XX quantile (XX determined by initial data)
	Quantile_df = A_W_df.groupby(['X_Target', 'Y_Target']).quantile(X_quantile, interpolation='lower')
	"""
	# Create a Dataframe with only the coordinates of warehouses and Tot_dist
	A_W_df = Selected_points_trgt_df[['X_Point','Y_Point','Tot_dist']]
	
	# Calculate the XX quantile (XX determined by initial data)
	Quantile_df = A_W_df.groupby(['X_Point','Y_Point']).quantile(X_quantile, interpolation='lower')
	
	#######################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	########################################################################################
	if Min_W < len(Proposed_Sites) < Max_W:
		q_mean = (Quantile_df.mean())
		fdist = q_mean[0]/60.0 # The mean of the quantiles of each asset. /60 = from seconds to minutes
	else:
		fdist = 100 # Meaningless very high number
	
	return fdist


def Calc_fdist_mean(Results_Folder, Proposed_Sites):
	# FUNCTION THAT CALCULATES fdist as the average TRavTime (in minutes) from a SINGLE proposed site to general/average target

	dist_dict_file = 'Dictionary_cells_targets'
	available_centroids_file = 'Available_centroids.shp'
	
	# print "Creating the dataframe from csv file..."
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	# sort data frame on targets' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True)

	# keeps the first row of each target (which contains the closest point)
	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True)

	agg_dist = Selected_points_trgt_df['Tot_dist'].sum() # sum of all the values of the column
	
	# print "aggregate dist = " , agg_dist
	# print "number of proposed sites for warehouse = " , len(Proposed_Sites)
	

	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	########################################################################################
	if len(Proposed_Sites) == 0:
		fdist = 10000 # Meaningless very high number
	else:
		# fdist = agg_dist/len(Proposed_Sites) # average sum (in seconds) of all the TRavTimes from a SINGLE proposed site to ALL the targets
		
		# COMPLETE FORMULA:
		# fdist = ((agg_dist/60)/len(Proposed_Sites))/(len(Selected_points_trgt_df.index)/len(Proposed_Sites)) # average TRavTime (in minutes) from a SINGLE proposed site to general/average target
		
		# SIMPLIFIED FORMULA:
		fdist = (agg_dist/60.0)/(len(Selected_points_trgt_df.index)) # average TRavTime (in minutes) from a SINGLE proposed site to general/average target
		
		# COMPONENTS OF THE FORMULA:
		# (agg_dist/60) = distance in minutes
		# (agg_dist/60)/len(Proposed_Sites) = average sum (in minutes) of all the TRavTimes from a SINGLE proposed site to ALL the targets
		# len(Selected_points_trgt_df.index) = lenght of the dataframe (= n_targets * n_proposed sites)
		# (len(Selected_points_trgt_df.index)/len(Proposed_Sites) = number of targets
		
	########################################################################################
	
	# print "Value of fdist = " , fdist
	# print
	return fdist


def Calc_fdist_GEUD(Results_Folder, Proposed_Sites, Min_W, Max_W, GEUD_power):
	# Function that calculates the travel time "à la Generalised Equivalent Uniform Dose" 
	# from each strategic infrastructure asset to the closest warehouse
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_avail_assets_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and Tot_dist
	# A_W_df = Selected_avail_assets_df[['X_Asset','Y_Asset','Tot_dist']]
	
	# Save travel times values into a list:
	# TT_var = Selected_avail_assets_df['Tot_dist']
	TT_list = Selected_avail_assets_df["Tot_dist"].tolist()
	
	GEUD = 0
	
	for tt in TT_list:
		GEUD = GEUD + tt**GEUD_power
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		fdist = GEUD**(1.0/GEUD_power)
	else:
		fdist = 999999 # Meaningless very high number	
	
	return fdist


def Calc_fdist_GEUD_checkDist(Results_Folder, Proposed_Sites, Min_W, Max_W, GEUD_power, Warehouse_Plan, Min_Warh_dist, Lookup):
	# Function that calculates the travel time "à la Generalised Equivalent Uniform Dose" 
	# from each strategic infrastructure asset to the closest warehouse
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets_Flood' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_avail_assets_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and Tot_dist
	# A_W_df = Selected_avail_assets_df[['X_Asset','Y_Asset','Tot_dist']]
	
	# Save travel times values into a list:
	# TT_var = Selected_avail_assets_df['Tot_dist']
	TT_list = Selected_avail_assets_df["Tot_dist"].tolist()
	
	GEUD = 0
	
	for tt in TT_list:
		GEUD = GEUD + tt**GEUD_power
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		fdist = GEUD**(1.0/GEUD_power)
	else:
		fdist = 500000 # Meaningless very high number
	
	# Check distance between warehouses:
	if Constraints.Check_Distance(Warehouse_Plan, Lookup, Min_Warh_dist) == True:
		# if warehouses are far enough from each other, do nothing
		pass
	else:
		fdist = 500000 # Meaningless very high number
		
	# for plot: reduce very big values
	if fdist > 500000.0:
		fdist = 500000.0
	
	return fdist


def Calc_fdist_squared(Results_Folder, Proposed_Sites, Min_W, Max_W):
	# Function that calculates the travel time "à la Generalised Equivalent Uniform Dose" 
	# from each strategic infrastructure asset to the closest warehouse
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_avail_assets_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and Tot_dist
	# A_W_df = Selected_avail_assets_df[['X_Asset','Y_Asset','Tot_dist']]
	
	# Save travel times values into a list:
	# TT_var = Selected_avail_assets_df['Tot_dist']
	TT_list = Selected_avail_assets_df["Tot_dist"].tolist()
	
	fdist = 0
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		for tt in TT_list:
			fdist = fdist + tt**2
	else:
		fdist = 999999999999 # Meaningless very high number	
	
	return fdist


def Calc_fcost(Proposed_Sites, Results_Folder):
	# Cost function that involves different sizes of warehouses
	
	# Dimension in square metres for Small, Medium and Large warehouses:
	S_W = 20*20  # squared metres (20mx20m)	- until 20 assets served
	M_W = 30*30  # squared metres (30mx30m) - until 60 assets served
	L_W = 40*50 # squared metres (40mx50m) 	- around 150 assets served
	
	Max_S_W = 20 # Maximum number of assets that a SMALL warehouse can serve
	Max_M_W = 60 # Maximum number of assets that a MEDIUM warehouse can serve
	
	average_cost  = 1.0 # unitary cost --> evaluating cost as space
	# average_cost  = 55.0 # average cost per warehouse = 55£ per sq meter per annum
	
	cost_S_W =  average_cost * S_W # annual average cost per SMALL warehouse
	cost_M_W =  average_cost * M_W # annual average cost per MEDIUM warehouse
	cost_L_W =  average_cost * L_W # annual average cost per LARGE warehouse
	
	# n_warehouses = len(Proposed_Sites) # number of warehouses
	
	# Determine how many assets are served by each warehouse:
	
	dist_dict_file = 'Dictionary_cells_targets'
	
	# Create the DataFrame from .csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a DataFrame containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)
	
	# Count how many assets are assigned to every proposed site:
	df = Selected_points_trgt_df.groupby(['X_Point','Y_Point']).size().reset_index(name="Served_Assets")
	
	# Save the number of served assets in a list:
	list_of_served_assets_numbers = df['Served_Assets'].values
	
	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	########################################################################################
	if len(Proposed_Sites) == 0:
		fcost = 100000 # Meaningless very high number
	else:
		fcost = 0
		for i in list_of_served_assets_numbers:
			if 0 < i <= Max_S_W:
				fcost = fcost + cost_S_W
			elif Max_S_W < i <= Max_M_W:
				fcost = fcost + cost_M_W
			elif Max_M_W < i:
				fcost = fcost + cost_L_W
	########################################################################################

	# print "number of warehouses = " , n_warehouses
	# print "annual rent price per warehouse = " , cost_per_cell
	# print "value of fcost = " , fcost
	return fcost


def Calc_fcost_dim(Proposed_Sites, Results_Folder, Data_Folder, Min_W, Max_W):
	# Cost function that involves different sizes of warehouses according to the amount of flood barriers needed
	
	# footprint of a 20ft. shipping container:
	# Container_area = 16.0 # Squared metres
	# 1 container = 100m of flood defences
	
	# Dimension in square metres for Small, Medium and Large warehouses:
	NF_W = 70.0  # squared metres - 4 containers - No flood defences, but only pumps and generators
	Small_W = 640.0  # squared metres - 40 containers = 4 km of flood barriers
	Medium_W = 1280.0 # squared metres - 80 containers = 8 km of flood barriers
	Big_W = 1920.0 # squared metres - 120 containers = 12 km of flood barriers
	Huge_W = 2560.0 # squared metres - 160 containers = 16 km of flood barriers
	
	Max_S_W = 4000.0 # Maximum temp flood def that a SMALL warehouse can store
	Max_M_W = 8000.0 # Maximum temp flood def that a MEDIUM warehouse can store
	Max_B_W = 12000.0 # Maximum temp flood def that a BIG warehouse can store
	
	average_cost  = 1.0 # unitary cost --> evaluating cost as space
	# average_cost  = 55.0 # average cost per warehouse = 55£ per sq meter per annum
	
	cost_NF_W =  average_cost * NF_W  # annual average cost per smallest warehouse
	cost_S_W =  average_cost * Small_W  # annual average cost per SMALL warehouse
	cost_M_W =  average_cost * Medium_W # annual average cost per MEDIUM warehouse
	cost_B_W =  average_cost * Big_W    # annual average cost per BIG warehouse
	cost_H_W =  average_cost * Huge_W   # annual average cost per HUGE warehouse
	
	# Determine how many assets are served by each warehouse and how much flood def lenght is required for their protection:
	dist_dict_file = 'Dictionary_cells_targets'
	
	# Create the DataFrame from .csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a DataFrame containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)
	
	# Create a df with the coordinates of the targets and the length of flood defencest they require
	strat_infr_file = 'Strat_infr_table.csv'
	
	# REMEBR to check that the table does NOT have headers (if it has headers, delete them!)
	
	strat_infr_table_df = pd.read_csv(Data_Folder+strat_infr_file, names=['ObjectID','Boolean','Typology','Orig_FID','In_floodzone','Temp_flood_def','X_Target','Y_Target'])
	columns = ['ObjectID','Boolean','Typology','Orig_FID','In_floodzone']
	strat_infr_table_df.drop(columns, inplace=True, axis=1) # Drop the columns that I don't need

	# Merge the dfs:	
	Pts_trgt_fldefs_df = Selected_points_trgt_df.merge(strat_infr_table_df, on=['X_Target','Y_Target'])

	# Count how many temporary defences are needed every proposed site:
	df = Pts_trgt_fldefs_df.groupby(['X_Point','Y_Point']).sum()

	# Save the amount of temporary defences lenght needed by the served assets in a list:
	list_of_needed_temp_def = df["Temp_flood_def"].tolist()
	
	# if len(list_of_needed_temp_def) == 0:
		# raise ValueError('List_of_needed_temp_def is an empty list. Even if no flood defs needed, the value should be 0, not empty. (Evaluate module, cost function)')
	
	# print list_of_needed_temp_def
	
	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	########################################################################################
	if Min_W < len(Proposed_Sites) < Max_W:
		fcost = 0
		for i in list_of_needed_temp_def:
			if i == 0:
				fcost = fcost + cost_NF_W
			elif 0 < i <= Max_S_W:
				fcost = fcost + cost_S_W
			elif Max_S_W < i <= Max_M_W:
				fcost = fcost + cost_M_W
			elif Max_M_W < i <= Max_B_W:
				fcost = fcost + cost_B_W
			elif Max_B_W < i :
				fcost = fcost + cost_H_W
	else:
		fcost = 10000 # Meaningless very high number
	########################################################################################

	# print "value of fcost = " , fcost
	return fcost


def Calc_fcost_constant(Proposed_Sites):
	# Constant fcost function
	# print "Calculate fcost function."
	
	average_cost  = 55.0 # average cost per warehouse = 55£ per sq meter per annum
	cell_dim	  = 2500.0 # squared meters
	warehouse_dim = 1000.0 # squared meters
	cost_per_cell =  average_cost * warehouse_dim # annual average cost per warehouse
	
	n_warehouses = len(Proposed_Sites) # number of warehouses:
	

	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	########################################################################################
	if len(Proposed_Sites) == 0:
		fcost = 300000 # Meaningless very high number
	else:
		fcost = n_warehouses * cost_per_cell
	########################################################################################

	# print "number of warehouses = " , n_warehouses
	# print "annual rent price per warehouse = " , cost_per_cell
	# print "value of fcost = " , fcost
	return fcost


# if __name__ == '__main__':
	
	# fdist = Calc_fdist(Results_Folder, Proposed_Sites)
	# fcost = Calc_fcost(Proposed_Sites)
	
# print	
# print "Program" , program_name, " started at: " , start_time
# end_time = time.asctime()
# print "Program" , program_name, " terminates at: " , end_time