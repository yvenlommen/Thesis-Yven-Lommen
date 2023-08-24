import pandas as pd
import numpy as np
import joblib
from constants import CUTOFF_NEGATIVE_DURATION_PERCENTAGE, NODE_CENTRALITIES, BINS
import networkx as nx
from operator import itemgetter
import copy
 

def largest_strongly_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """Return largest strongly connected component."""
    gc = G.subgraph(max(nx.strongly_connected_components(G), key=len))
    return gc.copy()

def create_network_graph(df: pd.DataFrame, id_column: str):
    """
    Create a network graph based on a dataset containing portcalls.
    Input:
        -df: dataframe with columns: "Arrival Time", "Departure Time", "Port Name", "IMO number", "Ship Type"
        -id_column: column that denotes the unique agents in the dataset
    Returns: 
        -G: nx.Digraph that has all the desired node attributes, which is the strongly connected component from the original graph
        -travel_times: all the values of travel between nodes
        -port_stay_times: all the values for the port stays
    """
    edges = []
    travel_times = np.array([]).astype(np.timedelta64)
    port_stay_times = np.array([]).astype(np.timedelta64)

    for name, group in df.groupby(id_column):
        #   Remove double entries of arrival and departure , this cannot be right
        group = group.drop_duplicates(subset="Arrival Time", keep='last')
        group = group.drop_duplicates(subset="Departure Time", keep='last')
        
        #   Discard if there is less then one entry (no journey can be made)
        if len(group) <= 1:
            continue

        #   calculate attributes for each ship
        weight = len(group) - 1
        distance = 1/(len(group) -1)
        group = group.sort_values("Arrival Time")
        group = group.reset_index(drop=True).dropna()
        source = group['Port Name'].shift(1).dropna().to_numpy()
        target = group['Port Name'].shift(-1).dropna().to_numpy()
        duration = (group["Arrival Time"] - group["Departure Time"].shift(1)).dropna().to_numpy()
        port_stay = group["Departure Time"] - group["Arrival Time"].dropna().to_numpy()

        #   sometimes, duration is < 0 because of an error in the data
        #   In this case we have selected a threshold for the maximum percentage of trips with negative durations
        #   if this threshold is reached, then we discard the entire ship
        #   Otherwise we keep the ship and it does not matter
        if np.any(duration, where= duration < pd.Timedelta(np.timedelta64(0, "ms"))):
            indices = np.argwhere(duration < pd.Timedelta(np.timedelta64(0, "ms")))
            indices = indices.reshape((indices.shape[0] * indices.shape[1]))
            negative_trips = len(indices)
            total_trips = len(duration)
            percentage_negative_trips = (negative_trips/total_trips) * 100
            if percentage_negative_trips > CUTOFF_NEGATIVE_DURATION_PERCENTAGE:
                continue
        df_group = pd.DataFrame({
                                'source': source,
                                'target': target,
                                'duration': duration,
                                'weight': weight,
                                'distance': distance,
                                })
        edges.append(df_group)
        travel_times = np.concatenate((travel_times,duration))
        port_stay_times = np.concatenate((port_stay_times, port_stay))

    # Create graph with attributes
    df_edges = pd.concat(edges, ignore_index=True)

    G = nx.from_pandas_edgelist(df_edges, edge_attr=True, 
                                    create_using=nx.DiGraph)    
    nx.set_node_attributes(G, dict(G.degree), 'degree')
    nx.set_node_attributes(G, dict(G.in_degree), 'in_degree')
    nx.set_node_attributes(G, dict(G.out_degree), 'out_degree')
    nx.set_node_attributes(G, dict(G.degree(weight='weight')), 'strength')
    nx.set_node_attributes(G, dict(G.in_degree(weight='weight')), 'in_strength'), 
    nx.set_node_attributes(G, dict(G.out_degree(weight='weight')), 'out_strength')
    nx.set_node_attributes(G, nx.closeness_centrality(G, wf_improved=False), 'closeness')
    nx.set_node_attributes(G, nx.closeness_centrality(G, distance='distance', wf_improved=False), 'closeness_weighted')
    nx.set_node_attributes(G, nx.betweenness_centrality(G, normalized=False), 'betweenness')
    nx.set_node_attributes(G, nx.betweenness_centrality(G, weight='weight', normalized=False), 'betweenness_weighted')
    nx.set_node_attributes(G, nx.eigenvector_centrality(G, max_iter=100_000), 'eigenvector')
    nx.set_node_attributes(G, nx.eigenvector_centrality(G, weight='weight', max_iter=100_000), 'eigenvector_weighted')

    G = largest_strongly_connected_component(G)

    return G, travel_times, port_stay_times, df_edges

def div(a,b):   
    """
    Divide function for our specific use case, because it handles 0 division and uses the maximum value obeserved
    from other operation outcomes (over the whole dataset).
    Input:
        -a/b: either a numeric value or array with numeric values
    returns: 
        -outcome: either a numeric value or an array (dependent on input)
    """ 
    # In our case the maximum value was 26 841 706.68522611
    max_value = 30000000
    
    # For arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        b_zero_indices = np.argwhere(b == 0)
        b_zero_indices = b_zero_indices.reshape((b_zero_indices.shape[0] * b_zero_indices.shape[1]))
        a_zero_indices = np.argwhere(a == 0)
        a_zero_indices = a_zero_indices.reshape((a_zero_indices.shape[0] * a_zero_indices.shape[1]))
        b_inf_indices = np.argwhere(b == float('inf'))
        b_inf_indices = b_inf_indices.reshape((b_inf_indices.shape[0] * b_inf_indices.shape[1]))
        a_inf_indices = np.argwhere(a == float('inf'))
        a_inf_indices = a_inf_indices.reshape((a_inf_indices.shape[0] * a_inf_indices.shape[1]))
        
        outcome = a / b
        outcome[b_zero_indices] = max_value
        outcome[a_zero_indices] = 0
        outcome[b_inf_indices] = 0
        outcome[a_inf_indices] = max_value
        return outcome

    # For integers/floats
    else:
        if a == 0:
            return 0       
        elif b == 0: 
            return max_value
        return a / b 



def create_feature_bins(G: nx.DiGraph, travel_times: np.array, port_stay_times: np.array):
    """
    Creates for every feature the boundaries for each bin based on percentiles.
    In order to prevent values occuring outside the bins, the outer boundaries are -inf and inf respectively.
    Input:
        -G: nx.DiGraph which contains the values for each node centrality measure
        -travel_times: all observed values for the travel time
        -port_stay_times: all observed values for port stay times
    returns: 
        -feature_bins: dictionary with for each feature the bin boundaries
    """
    feature_values = { 'travel_times': travel_times.astype(np.float64), 
                        'port_stay_times': port_stay_times.astype(np.float64)
                    }

    centrality_operations = {
                            'sub': np.subtract, 
                            'add': np.add, 
                            'mul': np.multiply, 
                            'div': div
                            }

    node_centralities = NODE_CENTRALITIES

    #   Retrieve all the values for each feature in order to do the binning
    #   A feature is a centrality measure/node attribute in combination with the operations
    edge_list = np.array(G.edges)
    sources = edge_list[:,:1].reshape((edge_list.shape[0]))
    targets = edge_list[:,1:].reshape((edge_list.shape[0]))

    for centrality in node_centralities:
        measures = nx.get_node_attributes(G, centrality)
        values_sources = np.array(itemgetter(*sources)(measures)).astype(np.float64)
        values_targets = np.array(itemgetter(*targets)(measures)).astype(np.float64)
        
        for operation in centrality_operations.keys():
            feature_name = centrality + "_" + operation
            outcome = centrality_operations[operation](values_sources, values_targets)
            feature_values[feature_name] = outcome

    # Create the bins for each feature
    feature_bins = {}
    for feature, values in feature_values.items():
        #   for the negative traveltimes we create a separate bin with negative values
        #   so we exclude these cases from the original binning
        if feature == "travel_times":
            values = values[values>=0]
        percentiles = np.percentile(values, [10,20,30,40,50,60,70,80,90])
        bins = np.array([float('-inf')])
        bins = np.concatenate((bins,percentiles))
        bins = np.concatenate((bins,[float('inf')]))
        feature_bins[feature] = bins

    return feature_bins

def digit_to_onehot(bin: int, number_of_bins: int):
    """
    Creates a one-hot representation of a bin
    Input:
        -bin: which bin is active
        -number_of_bins: amount of bins
    returns: 
        -encoding: one-hot representation of bin, example with 10 bins and bin number 2 = [0,1,0,0,0,0,0,0,0,0]
    """
    encoding = np.zeros(number_of_bins)
    encoding[bin - 1] = 1
    return encoding 

def array_to_onehot(bin_array: np.array, number_of_bins: int):
    """
    Creates a one-hot representation of a bin for an array. 
    Input:
        -bin_array: all the bins that need to be represented as one-hot
        -number_of_bins: amount of bins
    returns: 
        -encoding: output with shape = (len(bin_array), number_of_bins)
    """
    encodings = np.zeros((bin_array.shape[0],number_of_bins))
    for i in range(bin_array.shape[0]):
        encodings[i][bin_array[i] - 1] = 1
    return encodings

def retrieve_journeys_ship(df: pd.DataFrame):
    """
    Retrieves all usable journeys from an agent, with attributes for each journey.
    For usage see get_feature_df function
    Input:
        -df: dataframe for ONLY one ship, this function assumes that all entries belong to the same ship
    returns: 
        -journeys: dictionary with attributes
    """
    #   Remove double entries of arrival and departure , this cannot be right
    df = df.drop_duplicates(subset="Arrival Time", keep='last')
    df = df.drop_duplicates(subset="Departure Time", keep='last')

    
    df = df.sort_values("Arrival Time")
    df = df.reset_index(drop=True).dropna()
    source = df['Port Name'].shift(1).dropna().to_numpy()
    target = df['Port Name'].shift(-1).dropna().to_numpy()
    duration = (df["Arrival Time"] - df["Departure Time"].shift(1)).dropna().to_numpy()
    port_stay = (df["Departure Time"] - df["Arrival Time"]).dropna().to_numpy()

    negative_trips = 0

    # discard ships that have no trips (or ports are NA)
    if len(source) ==0 or len(target) == 0:
        return 0
 
    #   discard the ship if the perecnetage of negative durations is higher than threshold
    if np.any(duration, where= duration < pd.Timedelta(np.timedelta64(0, "ms"))):
        indices = np.argwhere(duration < pd.Timedelta(np.timedelta64(0, "ms")))
        indices = indices.reshape((indices.shape[0] * indices.shape[1]))
        negative_trips = len(indices)
        total_trips = len(duration)
        percentage_negative_trips = (negative_trips/total_trips) * 100
        if percentage_negative_trips > CUTOFF_NEGATIVE_DURATION_PERCENTAGE:
            # return 0 if we discard the ship
            return 0

    journeys = {
                'source': source,
                'target': target,
                'travel_times': duration.astype(np.float64),
                'port_stay_times': port_stay.astype(np.float64),
                'negative_travel_times': negative_trips
                 }
    return journeys

def get_feature_df(df: pd.DataFrame, G: nx.digraph, feature_bins: dict, id_column: str, target_column: str):
    """
    Generates the dataframe that can be used for classification, so all the independent variables and the target
    Input:
        -df: pd.DataFrame which contains all the portcalls of all the ships we want to classify
        -G: Graph which is used for feature engineering
        -feature_bins: boundaries for each feature
        -id_column: identifier for ships
        -target_column: column with the target for the classification task
    returns: 
        -df_features: pd.DataFrame with all the relevant variables, target variable column is now labeled 'Target'
    """
    centrality_operations = {
                                'sub': np.subtract, 
                                'add': np.add, 
                                'mul': np.multiply, 
                                'div': div
                            }

    node_centralities = NODE_CENTRALITIES

    # Create placeholder for base values 
    template_features = {'one_port_missing': 0,
                        'two_ports_missing': 0,
                        'amount_of_journeys': 0,
                        'negative_travel_times': 0,
                        'port_stay_times': [],
                        'travel_times': [] 
                        }

    features_to_bin = ['port_stay_times', 'travel_times']
    for centrality in node_centralities:
        for operation in centrality_operations.keys():
            template_features[centrality + "_" + operation] = []
            features_to_bin.append(centrality + "_" + operation)

    ship_count = 0 
    features = []
    #   Create feature values for each ship
    for name, group in df.groupby(id_column):
        feature_values = copy.deepcopy(template_features)
        journeys = retrieve_journeys_ship(group)
        
        if journeys != 0:
            #   Retrieve feature values for each journey
            feature_values['IMO'] = name
            feature_values['Target'] = group[target_column].unique()[0]
            ship_count += 1
            feature_values['negative_travel_times'] = journeys['negative_travel_times']
            feature_values['port_stay_times'] = journeys['port_stay_times']
            feature_values['travel_times'] = journeys['travel_times'][journeys['travel_times'] >= 0]
            for source, target in zip(journeys['source'], journeys['target']):
                if (source not in G.nodes) and (target not in G.nodes):
                    feature_values['two_ports_missing'] += 1 # GJ also divides this by number of journeys? should I also do this
                elif (source not in G.nodes) or (target not in G.nodes):
                    feature_values['one_port_missing'] += 1
                else:
                    feature_values['amount_of_journeys'] += 1
                    for centrality in node_centralities:
                        for operation in centrality_operations.keys():
                            feature_name = centrality + "_" + operation
                            value = centrality_operations[operation](G.nodes[source][centrality], G.nodes[target][centrality])
                            feature_values[feature_name].append(value)


            #   Bin each feature value and transform to one-hot format
            for feature in features_to_bin:
                bins = feature_bins[feature]
                values = feature_values[feature]
                binned_values = np.digitize(values,bins)
                dummy_variable_names = []

                #   Create one extra bin for travel times because of negative ones
                if feature == 'travel_times':
                    encodings = array_to_onehot(binned_values, BINS + 1)
                    for i in range(BINS + 1):
                        dummy_variable_names.append(feature + "_dummy_" + str(i))
                else: 
                    encodings = array_to_onehot(binned_values, BINS)
                    for i in range(BINS):
                        dummy_variable_names.append(feature + "_dummy_" + str(i))
                
                #   Sum and normalize the bin values such that each array with values sums to one
                encodings = np.sum(encodings, axis=0)
            
                #   There is always one more port stay than travels
                if feature == 'port_stay_times':
                    encodings = encodings / len(feature_values['port_stay_times'])
                
                #   Fill the extra bin for travel times
                elif feature == 'travel_times':
                    encodings[BINS] = feature_values['negative_travel_times'] 
                    encodings = encodings / (len(feature_values['travel_times']) + feature_values['negative_travel_times'])
                    del feature_values['negative_travel_times']
                else:
                    encodings = encodings / feature_values['amount_of_journeys']
                    
                #   happens when one or two ports are missing 
                encodings[np.isnan(encodings)] = 0.

                del feature_values[feature]
                dummy_feature_values = dict(zip(dummy_variable_names, encodings))
                feature_values.update(dummy_feature_values)
            
            df_group = pd.DataFrame(feature_values, index=[ship_count])
            features.append(df_group)

    df_features = pd.concat(features, ignore_index=True)
    return df_features



