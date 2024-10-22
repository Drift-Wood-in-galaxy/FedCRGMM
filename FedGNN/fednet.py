from model import *
    
class FederatedNetwork:
    
    def __init__(self, initial_weights, device):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = device
#         self.preivous_weights=None
        self.initialize_model(initial_weights)
        
    def initialize_model(self, initial_weights):
        self.model = GAT().to(self.device)
        if initial_weights != None:
            self.model.load_state_dict(initial_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=5e-4)
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=0.9)

        self.criterion = nn.MSELoss()  #Square Root taken later in training to make RMSE
        


        