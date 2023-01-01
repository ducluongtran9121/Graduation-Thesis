from threading import local
import numpy
from .models import AE
from .utils import *
from .CKA import CKA
from sklearn.cluster import KMeans

class Defense_AE():
    def __init__(self) -> None:
        self.model = AE()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 10
        self.threshold = 1.0
        self.cluster_Kmeans = KMeans(n_clusters=2, random_state=0)
    
    def train(self, local_models):
        print('[*] Pretrain anti-poisoning model - AE ...')
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for data in local_models:
                data = torch.from_numpy(data)
                output = self.model(data)
                loss = self.criterion(output, data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.data
            print('\tepoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, total_loss/len(local_models)))
        torch.save(self.model.state_dict(), f'/home/haochu/Documents/Federated-Averaging-PyTorch/defense/AE_e{self.epochs}.pth')
    
    def test_IO(self, local_models, sampled_client_indices):
        print('[*] Detect poisoning model stage by AE ...')
        poisoning_idx = {}
        self.model.eval()
        idx = 0
        for data in local_models:
            data = torch.from_numpy(data)
            output = self.model(data)
            loss = self.criterion(output, data)
            if loss.item() > self.threshold:
                poisoning_idx[sampled_client_indices[idx]] = 1
            else:
                poisoning_idx[sampled_client_indices[idx]] = 0
            idx += 1
            print('\tLocal Model [{}/{}], loss:{:.4f}'.format(idx, len(local_models), loss.item()))
        return poisoning_idx

    def test_latentspace(self, global_plr, local_models, sampled_client_indices):
        print('[*] Detect poisoning model stage by AE ...')
        self.model.eval()
        np_cka = CKA()
        local_cka = np.array([])
        global_latentspace = self.model.encode(torch.from_numpy(global_plr)).detach().numpy()
        
        for it, data in enumerate(local_models,1):
            data = torch.from_numpy(data)
            local_latentspace = self.model.encode(data).detach().numpy()
            cka = np_cka.kernel_CKA(local_latentspace, global_latentspace)
            local_cka = np.append(local_cka, [cka], axis=0)
            print('[-] {}: RBF Kernel CKA, between local LS {} and global LS: {}'.format(it, it, cka))

        local_cka = local_cka.reshape(len(local_models), 1)
        kmeans = self.cluster_Kmeans.fit(local_cka)
        y_pred = kmeans.predict(local_cka)
        poisoning_idx = {}
        for idx, is_poisoning in enumerate(y_pred):
            poisoning_idx[sampled_client_indices[idx]] = is_poisoning
        return poisoning_idx