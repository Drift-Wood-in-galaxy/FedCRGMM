from fednet import *
from Setting import *
import random
class Client(FederatedNetwork):
   
    def __init__(self, client_id, items_rated_df, all_items_embeddings_df, initial_weights, device):
        super().__init__(initial_weights, device)
        self.id = client_id
        
        self.items_rated_df = items_rated_df
        self.all_items_embeddings_df = all_items_embeddings_df
        self.rated_items_embeddings_df = None
        self.user_embeddings = None
        
        self.train_idx = None
        self.val_idx = None
        
        self.train_graph = None
        self.train_y = None
        self.val_graph = None
        self.val_y = None
        
        self.train_x_df = None
        self.valid_x_df = None
        self.train_y_df = None
        self.valid_y_df = None
        
        
        self.neighbors = None
        
        self.initialize_client()
        
        
    def find_splits(self):
        data_len = len(self.items_rated_df)
        data_idx = random.sample(range(0, data_len), 
                                 data_len)
        train_idx = data_idx[round(data_len*VALIDATION_RATIO):]
        valid_idx = data_idx[:round(data_len*VALIDATION_RATIO)]
        

        return train_idx, valid_idx
        
        
    def initialize_client(self):
        self.user_embeddings = torch.nn.init.xavier_uniform_(torch.empty(USER_EMBEDDING_SIZE, 256))
        self.train_idx, self.valid_idx = self.find_splits()
        
        self.split_data_and_create_graphs()
        
    
    def split_data_and_create_graphs(self):
        
        self.rated_items_embeddings_df = self.all_items_embeddings_df[
            self.all_items_embeddings_df['movieId'].isin(self.items_rated_df["movieId"])]
        
        self.train_x_df = self.rated_items_embeddings_df.iloc[self.train_idx]
#         if self.valid_x_df == None:
        self.valid_x_df = self.rated_items_embeddings_df.iloc[self.valid_idx]
        self.train_y_df = self.items_rated_df.iloc[self.train_idx]
        self.valid_y_df = self.items_rated_df.iloc[self.valid_idx]
        
        self.create_graph_data()
    
        
    def create_graph_data(self):
        x, y, edge_index = self.convert_df_to_graph(self.train_y_df, self.train_x_df)
        self.train_graph = Data(x=x, edge_index=edge_index).to(self.device)
        self.train_y = y
        
        x, y, edge_index = self.convert_df_to_graph(self.valid_y_df, self.valid_x_df)
        self.val_graph = Data(x=x, edge_index=edge_index).to(self.device)
        self.val_y = y
        
        
    def convert_df_to_graph(self, items_rated_df, rated_items_embeddings_df):
        
        edges_start = [0]*(len(items_rated_df)) + [i for i in range(1, len(items_rated_df)+1)]
        edges_end = [i for i in range(1, len(items_rated_df)+1)]+[0]*(len(items_rated_df)) 
        
        if self.neighbors != None and INCLUDE_NEIGHBORS:
            edges_start += [0]*(len(self.neighbors)) + [i for i in range(len(items_rated_df)+1, 
                                                            len(items_rated_df)+1+len(self.neighbors))]
            edges_end += [i for i in range(len(items_rated_df)+1,
                                len(items_rated_df)+1+len(self.neighbors))] + [0]*(len(self.neighbors))
        
        edge_index = torch.tensor([edges_start, edges_end], dtype=torch.long)
        
        x = [self.user_embeddings.numpy()[0], ] #User Embeddings
        embeddings_col = rated_items_embeddings_df['embeddings'].values
        x += [val for val in embeddings_col]   #Item Embeddings

        if self.neighbors != None and INCLUDE_NEIGHBORS:
            x += self.neighbors
            
        x = torch.tensor(np.array(x), dtype=torch.float) #Converting embeddings array into tensor
        
        y = torch.tensor(items_rated_df['rating'].values)
        
        return x, y, edge_index
    
    
    def update_to_global_weights(self, weights):
        self.model.load_state_dict(weights)
    
    
    def train_model(self, lr=LR):
        self.model.train()
        for epoch in range(EPOCHS):
            self.optimizer.zero_grad()
            x, out = self.model(self.train_graph, len(self.train_x_df)+1)
            loss = torch.sqrt(self.criterion(out, self.train_y.to(self.device)))            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()  
            self.train_graph.x -= lr*x.grad
            
        self.update_df_embeddings_from_train_graph()
        
        return loss
    
    def train_user_embeddings(self, lr=LR):
        self.model.train()
        for epoch in range(EPOCHS):
            x, out = self.model(self.train_graph, len(self.train_x_df)+1)
            loss = torch.sqrt(self.criterion(out, self.train_y.to(self.device)))            
            loss.backward()
            self.train_graph.x[0] -= lr*x.grad[0]
#             print(self.train_graph.x[0])

        self.user_embedding = self.train_graph.x[0]

        return loss
    
    def get_item_ids(self):
        return self.rated_items_embeddings_df['movieId'].values
    
    
    def update_df_embeddings_from_train_graph(self):
        self.user_embedding = self.train_graph.x[0]
        self.val_graph.x[0] = self.train_graph.x[0]
        for i in range(1, len(self.train_x_df)):
            index = self.all_items_embeddings_df.index[
                self.all_items_embeddings_df['movieId'] == self.rated_items_embeddings_df.iloc[i-1]['movieId']]
            self.rated_items_embeddings_df.iat[i-1, 1] =  self.train_graph.x[i]
            self.all_items_embeddings_df.iat[index[0], 1] =  torch.Tensor.cpu(self.train_graph.x[i]).numpy()
            
            
    def get_item_embeddings(self):
        return self.all_items_embeddings_df
    
        
    def item_count(self):
        return len(self.rated_items_embeddings_df)
    
    
    def update_to_global_embeddings(self, items_embeddings):
        self.all_items_embeddings_df = items_embeddings
        self.split_data_and_create_graphs()
    
    
    def evaluate_model(self, model=None):
        data = self.val_graph

        if model == None:
            model = self.model

        model.eval()
        
        _, pred = model(data, len(self.valid_x_df)+1)
        pred = torch.round(2*pred.data)/2
        
        loss = torch.sqrt(self.criterion(pred, self.val_y.to(self.device)))  

        #AUC evaluation
        correct = float(pred.eq(self.val_y.to(self.device)).sum().item())
        acc = correct / len(self.val_y.to(self.device))
        
        #RMSD evaluation
        #RMSD = None
        #RMSD = np.linalg.norm((self.val_y - torch.Tensor.cpu(pred).numpy()),axis = None,keepdims=False) / np.sqrt(len(self.val_y))
            

        return acc, loss
    
    def include_neighbors(self, neighbors_embeddings):
        self.neighbors = neighbors_embeddings