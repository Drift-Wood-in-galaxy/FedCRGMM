from Myplt import *
from initailizer import *
from Setting import *
from utils import *
import random

def train(global_model,initializer):

    # test_acc_l = []
    # test_loss_l = []
    # for client in test_clients:
    #     client.model.load_state_dict(global_model.state_dict())
    #     test_acc, test_loss = client.evaluate_model()
    #     test_acc_l.append(test_acc)
    #     test_loss_l.append(test_loss)
    # print("Initial:\nAccuracy: ", torch.mean(torch.Tensor(test_acc_l)).numpy(), "\nLoss: ", torch.mean(torch.Tensor(test_loss_l)).numpy())

    lost = []
    train_lost=[]

    for training_round in range(ROUNDS):
        total_items = 0
        weights = []
        embeddings = [] 
        items_rated = []
        losses = []
        
        if training_round == NEIGHBORS_THRESHOLD:
            initializer.include_neighbors_embeddings()
        
        selected_clients = random.sample(range(0, len(initializer.clients)), ROUND_CLIENTS)
        
        #TRAIN THE NETWORK
        for client_idx in selected_clients:
            losses.append(initializer.clients[client_idx].train_model())
            total_items += initializer.clients[client_idx].item_count()
            weights.append(initializer.clients[client_idx].model.state_dict())
            embeddings.append(initializer.clients[client_idx].get_item_embeddings())
            items_rated.append(initializer.clients[client_idx].item_count())
        
        #UPDATE GLOBAL MODEL

        #WEIGHTED AVERAGE: 
        new_parameters = global_model.state_dict()
        for key in new_parameters:
            new_parameters[key] = weights[0][key]
            for i in range(1, len(weights)):
                new_parameters[key] += weights[i][key]*(items_rated[i])
            new_parameters[key]/=float(total_items)
            
        global_model.load_state_dict(new_parameters)
        
        #UPDATE GLOBAL EMBEDDINGS    
        #df = pd.concat(laplace_mech(i,Lambda,1) for i in embeddings)
        df = pd.concat(i for i in embeddings)
        global_embeddings = df.groupby(by="movieId", as_index=False).mean()
        global_embeddings.reset_index()
        
        # EVALUATE
        acc = 0
        l_acc = 0
        g_loss = 0
        l_loss = 0
        for client in initializer.clients:
            acc_temp, g_loss_temp = client.evaluate_model(global_model)
            l_acc_temp, l_loss_temp = client.evaluate_model()
            acc += acc_temp
            g_loss += g_loss_temp
            l_acc += l_acc_temp
            l_loss += l_loss_temp
        loss = 0
        for i in losses:
            loss += i
        loss/=ROUND_CLIENTS
        
        
        print("\n\nRound: ", training_round,file=open("record.txt","a"))
        print('Validation Accuracy [Global Model][All Clients]: {:.4f}'.format(acc/ROUND_CLIENTS),file=open("record.txt","a"))
        print('Validation Accuracy [Local Model][All Clients]:  {:.4f}'.format(l_acc/ROUND_CLIENTS),file=open("record.txt","a"))

        print('Training Loss [Local Model][Train Clients]: {:.4f}'.format(loss.item()),file=open("record.txt","a"))
        print('Validation Loss [Global Model][All Clients]: {:.4f}'.format(g_loss.item()/ROUND_CLIENTS),file=open("record.txt","a"))
    #     print('Local Valid Loss: {:.4f}'.format(l_loss.item()/ROUND_CLIENTS))
        lost.append(g_loss.item()/ROUND_CLIENTS)
        train_lost.append(loss.item())


        initializer.items_embeddings_df = global_embeddings
        # UPDATE LOCAL CLIENTS
        for client in initializer.clients:
            client.update_to_global_weights(global_model.state_dict())
            client.update_to_global_embeddings(global_embeddings)
            
    train_plt(lost,train_lost)
    return initializer
    

            