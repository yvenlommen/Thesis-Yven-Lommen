import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from constants import START_DATE, END_DATE

def select_columns(df: pd.DataFrame, relevant_columns: dict, new_column_names: list):
    """
    Function that transforms columns to the desired datatype and renames these columns.
    Inputs:
        -relevant columns: dictionary with (key,value) pairs which represent (column name, dtype)
        -new column names: list with the new column names which replace the old column names
    """  
    #   retrieve current column names and transform to dict with new names
    current_column_names = list(relevant_columns.keys())
    replace_columns = dict(zip(current_column_names, new_column_names))


    #   Only keep the relevant columns and transform the type
    df = df[current_column_names]
    df = df.astype(relevant_columns)
    df = df.rename(columns=replace_columns)

    return df

def get_time_frame_data(df: pd.DataFrame, start_date: datetime.datetime, end_date: datetime.datetime, date_column: str):
    """
    Get all the data within the specified dates, both start and end dates are included in the subset
    """
    return df.loc[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

def remove_sinlge_occurences(df: pd.DataFrame, column_name: str):
    """
    Removes agents that only have one occurence in the data under the specified column.
    """
    return df.loc[df.duplicated(subset=column_name, keep=False)]

def split_network_train_set(df: pd.DataFrame, id_column: str, label_column: str, network_size=0.1):
    """
    splits the dataset in a stratified manner into a training set and network set.  
    Inputs:
        -ID column: column which to use to uniquely identify agents
        -label column: column which is used to label an agents
    """
    df_original = df

    #   Drop all the duplicates and agents that have no labels.
    df = df.drop_duplicates([id_column])
    df = df.dropna(subset=[label_column])


    agent_ids = df[id_column].to_numpy()
    agent_labels = df[label_column].to_numpy()

    #   Retrieve a list of unique labels
    labels = pd.unique(df[label_column])

    #   Retrieve numeric labels
    stratify = []

    for agent in agent_labels:
        index = np.where(labels == agent)
        stratify.append(index[0][0])

    stratify = np.array(stratify)

    agents_classify, agents_network = train_test_split(agent_ids, test_size=network_size, random_state=42,stratify=stratify)

    df_network = df_original.loc[df_original[id_column].isin(agents_network)]
    df_classify = df_original.loc[df_original[id_column].isin(agents_classify)]

    return df_network, df_classify

def keep_classes(df: pd.DataFrame, classes_to_keep: list, target_column: str, id_column: str):
    """
        Preprocessing step such that all agents with no label are discarded, ensures label consistency for unique agents, 
        and only keeps the desired classes (all other classes are labeled 'Other')
        Input:
            -classes_to_keep: list with all the class names we want to keep
            -target_column: column name of the target label
            -id_column: column name for the unique identification of agents
    """
    agents_to_discard = []
    agent_type_dict = {}
    
    for name, group in df.groupby(id_column):
        target_classes = group[target_column].dropna().to_numpy()
        #   If an agent only has NAN values in its entries, we discard it
        if np.any(target_classes) == False:
            agents_to_discard.append(name)
        #   otherwise its label becomes the first entry we see
        else: 
            agent_type_dict[name] = target_classes[0]
            
    #   discard agents with no label
    
    df = df.loc[~df[id_column].isin(agents_to_discard)]
    
    #   Assure that labels for unique agents are consistent
    df.loc[:, target_column] = df[id_column].map(lambda x: agent_type_dict[x])

    #   transform every irrelevant classname to 'other'
    df.loc[~df[target_column].isin(classes_to_keep), target_column] = 'Other'
    
    return df.reset_index(drop=True)



def get_subset_exclude_ids(df: pd.DataFrame, exclude_ids: np.array, id_column: str, time_column: str,
                               start_date: datetime.datetime, end_date: datetime.datetime):
    """
    Returns a subset with start and end date without the IDs occuring in the list of ids to exclude.
    A subset can be used for either train or test. Used for gap experiment.
    Input:
        exclude_ids: np.array continaing all ids that need to be excluded from the subset
    """
    # retrieve the ids that occur in the subset
    df_subset = get_time_frame_data(df, start_date, end_date, time_column)

    subset_ids = pd.unique(df_subset[id_column])

    overlap = np.intersect1d(exclude_ids, subset_ids)

    # retrieve the unique values in subset and excluding ids
    subset_ids = subset_ids[np.isin(subset_ids, overlap, invert=True)]

    #   assign right timeframe
    df_subset = get_time_frame_data(df, start_date, end_date, time_column)

    #   only keep the right instances
    df_subset = df_subset.loc[df_subset[id_column].isin(subset_ids)]

    return df_subset

def get_subset_include_ids(df: pd.DataFrame, include_ids: np.array, id_column: str, time_column: str,
                               start_date: datetime.datetime, end_date: datetime.datetime):
    """
    Returns a subset of the data with only the ids that are provided. Used for expanding encoding and gap
    """
    subset = get_time_frame_data(df, start_date, end_date, time_column)

    return subset.loc[subset[id_column].isin(include_ids)]
