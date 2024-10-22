from client import *
import numpy as np
import pandas as pd
from Setting import *
from model import *
import random

class Initializer:

    def __init__(self, device, initial_weights):
        
        #2-col df_items_embeddings with movieId and embeddings
        self.items_embeddings_df = None
        
        #all columns from ratings.csv
        self.items_ratings_df = None
        
        #all columns from movies.csv file
        self.items_info_df = None        
        self.clients = None

        self.train_clients_idx = None
        self.test_clients_idx = None
        self.initial_weights = initial_weights
        
        self.device = device
        self.generate_items_embeddings()
        self.initialize_clients()
        
        self.client_user_embeddings = None

    def initialize_clients(self):
        self.items_ratings_df = pd.read_csv(RATINGS_DATAFILE)
        self.items_ratings_df.drop('timestamp', inplace=True, axis=1)
        
        # RANDOMLY SELECTING TEST/TRAIN CLIENTS
        clients_idx =  random.sample(range(1, TOTAL_CLIENTS+1), TOTAL_CLIENTS)
        split_index = round(TOTAL_CLIENTS*TRAIN_CLIENTS_RATIO)
        self.train_clients_idx = clients_idx[:split_index]
        self.test_clients_idx = clients_idx[split_index:]
        self.clients = [Client(client_id=user_idx, 
                              items_rated_df=self.items_ratings_df[self.items_ratings_df['userId'] == user_idx], 
                              all_items_embeddings_df = self.items_embeddings_df, 
                              initial_weights=self.initial_weights,
                              device = self.device) for user_idx in self.train_clients_idx]
                        
        return self.clients
    
    def initialize_clients_for_induction(self):
        return [Client(client_id=user_idx, 
                              items_rated_df=self.items_ratings_df[self.items_ratings_df['userId'] == user_idx], 
                              all_items_embeddings_df = self.items_embeddings_df, 
                              initial_weights=self.initial_weights,
                              device = self.device) for user_idx in self.test_clients_idx]
    
        
    def generate_items_embeddings(self):
        self.items_info_df = pd.read_csv(MOVIES_INFO_DATAFILE)
        
        # Initialize weights with xavier uniform
        embeddings = torch.nn.init.xavier_uniform_(torch.empty(self.items_info_df.shape[0], 256))
        
        # Creating 1-col df for embeddings
        df = pd.DataFrame({"id": np.arange(1, embeddings.shape[0]+1)})
        df["embeddings"] = list(embeddings.numpy())
        
        # Creating 2-col df items_embeddings with movieId and embeddings
        self.items_embeddings_df = pd.concat([self.items_info_df['movieId'], df["embeddings"]], axis=1)
    
    def get_items_embeddings(self):
        return self.get_item_embeddings_df
    
    
    def include_neighbors_embeddings(self):
            
        neighbor_user_embeddings_dict = dict()
        for client_index1 in range(len(self.clients)):
            item_ids = self.clients[client_index1].get_item_ids()
            for client_index2 in range(client_index1+1, len(self.clients)):
                item_ids_2 = self.clients[client_index2].get_item_ids()
                for item_id in item_ids:
                    if item_id in item_ids_2:
                        if client_index1 in neighbor_user_embeddings_dict:
                            neighbor_user_embeddings_dict[client_index1] += [
                                self.clients[client_index2].user_embeddings.numpy()[0]]
                        else:
                            neighbor_user_embeddings_dict[client_index1] = [
                                self.clients[client_index2].user_embeddings.numpy()[0]]
                        if client_index2 in neighbor_user_embeddings_dict:
                            neighbor_user_embeddings_dict[client_index2] += [
                                self.clients[client_index1].user_embeddings.numpy()[0]]
                        else:
                            neighbor_user_embeddings_dict[client_index2] = [
                                self.clients[client_index1].user_embeddings.numpy()[0]]
                        break
                    
        for client_id in neighbor_user_embeddings_dict:
            self.clients[client_id].include_neighbors(neighbor_user_embeddings_dict[client_id])