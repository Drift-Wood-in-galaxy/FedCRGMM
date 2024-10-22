from train import *
from model import *
from initailizer import *
from Myplt import *
from Setting import *
    
def test(global_model,test_clients):
    test_loss = []
    INDUCTIVE_ROUNDS = 25
    for i in range(INDUCTIVE_ROUNDS):
        test_loss.append(np.array([torch.Tensor.cpu(test_client.train_user_embeddings().detach()).numpy() for test_client in test_clients]).mean())
    test_plt(test_loss,INDUCTIVE_ROUNDS)
    
    test_acc_l = []
    test_loss_l = []
    for client in test_clients:
        client.model.load_state_dict(global_model.state_dict())
        test_acc, test_loss = client.evaluate_model()
        test_acc_l.append(test_acc)
        test_loss_l.append(test_loss)
    print("Final:\nAccuracy: ", torch.mean(torch.Tensor(test_acc_l)).numpy(), "\nLoss: ", torch.mean(torch.Tensor(test_loss_l)).numpy())

        
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = GAT().to(device)
    initializer = Initializer(device=device, initial_weights=global_model.state_dict())
    #print(initializer.clients[0].train_graph)
    
    initializer = train(global_model,initializer)
    
    test_clients = initializer.initialize_clients_for_induction()
    test(global_model,test_clients)
       
if __name__ == '__main__':
    main()