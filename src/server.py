import copy
import gc
from itertools import chain
import logging
from matplotlib.pyplot import contour

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from pyod.models.sos import SOS

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter, OrderedDict

from .models import *
from .utils import *
from .client import Client
from .defense import Defense_AE

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}, average_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model = eval(model_config["name"])(**model_config)
        
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]
        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]
        self.lr_server = average_config["lr_server"]
        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.defence_mode = fed_config["defence_mode"]
        self.attack_mode = fed_config["attack_mode"]
        self.scale_attack = fed_config["scale_attack"]
        self.optim_config = optim_config
        self.AE = Defense_AE()

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        local_datasets, test_dataset = create_datasets(self.dataset_name, self.num_clients, self.iid, self.attack_mode)
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
        
        # send the model skeleton to all clients
        self.transmit_model() 
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())
        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        selected_total_size = 0
        if self.defence_mode == 'AE':
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                if 1 <= self._round <= 2 and 1 <= it <= 4:
                    continue
                self.clients[idx].client_update()
                selected_total_size += len(self.clients[idx])
        else:
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].client_update()
                selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def defense_lof(self):
        pass

    def defence_det(self, w, d_out):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                if d_out[i]==0:
                    w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], (len(d_out)-sum(d_out)))
        return w_avg

    def defence_our(self, omega_locals, w_locals, w_local_pre):
        X_norm = []
        selected_index = {}
        for i in omega_locals[0].keys():
            aggregate_index = list()
            for j in range(0, len(omega_locals)):
                aggregate_index.append(omega_locals[j][i])
            selected_index[i] = Counter(list(chain(*aggregate_index)))
        # print('interation', interation, 'client', client)
        # print('selected_index', selected_index)
        # print("w: ", w_locals)s
        for i in range(0, len(w_locals)):
            selected_weights = []
            all_weights = []
            # print("client ", i)
            for n in w_locals[0].keys():
                # print(n)
                c = w_locals[i][n].cpu()
                c_pre = w_local_pre[n].cpu()
                # all_weights.append((c.view(-1).detach().numpy() - c_pre.view(-1).detach().numpy()))
                try:
                    selected_index_dict = dict(selected_index[n])
                except:
                    pass
                indice = []
                for a in range(0, len(selected_index_dict)):
                    # print("a", list(selected_index_dict.values())[a])
                    # if (list(selected_index_dict.values())[a] < 45)&(list(selected_index_dict.values())[a] >40):
                    # if (list(selected_index_dict.values())[a] > 90):
                        # if ((list(selected_index_dict.values())[a] > 30) & (list(selected_index_dict.values())[a] < 40)): # | (list(selected_index_dict.values())[a] > 95)
                    indice.append(list(selected_index_dict.keys())[a])     
                # print("indice: ",indice)
                # print("c: ", len(c.view(-1)))
                if (len(c.view(-1)) == 1):
                    # print("SOSSSS: ",n)
                    continue
                # print("C_pr: ", len(c_pre.view(-1)))
                if len(indice) > 0:
                    indices = torch.tensor(indice)
                    # print('indices', indices)
                    d = torch.index_select(c.view(-1), 0, indices)
                    d_pre = torch.index_select(c_pre.view(-1), 0, indices)
                    selected_weights.append((d.view(-1).detach().numpy() - d_pre.view(-1).detach().numpy()))
                else:
                    pass
            # print("selected_weights: ",selected_weights)
            X_norm.append(list(chain(*selected_weights)))
            # X_all.append(list(chain(*all_weights)))
            # print(selected_weights)
            # print(np.array(selected_weights))

        X_norm = np.array(X_norm)
        # file = "output_" + str(self._round) + ".npy"
        # np.save(file, X_norm)
        # self.writer.add_scalars("SOS: ", str(X_norm))
        # X_all = np.array(X_all)
        # print('X', X_norm.shape)
        #### OUR
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler.fit(X_norm)
        X_norm = scaler.transform(X_norm)
        # outliers_fraction = float(num_poison_client/num_clients)
        # print('######outliers_fraction',outliers_fraction)
        # random_state = 42
        # print("SOS: ",X_norm)
        clf = SOS(contamination=0.35, perplexity=self.num_clients-1)

        clf.fit(X_norm)
        pre_out_label = clf.labels_

        # print('prediction', pre_out_label)
        # print(confusion_matrix(Y_norm.astype(int), pre_out_label))
        # print(classification_report(Y_norm.astype(int), pre_out_label))
        # print("train AC", accuracy_score(Y_norm.astype(int), pre_out_label))
        ### Federated aggregation with defence
        print("Detected: ", pre_out_label)
        logging.info(pre_out_label)
        w_glob = self.defence_det(w_locals, pre_out_label)
        # self.model.load_state_dict(w_glob)

        return w_glob

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        omega_locals = []
        w_locals = []
        w_local_pre = self.model.state_dict()
        averaged_weights = OrderedDict()
        
        if self.defence_mode in ['SOS']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                # self.clients[idx].client_evaluate()
                omega_locals.append(self.clients[idx].omega_index)
                w_locals.append(self.clients[idx].model.state_dict())
                # model_name = "model_attack_{}_{}/Round{}_client{}".format(self.num_clients, self.attack_mode, self._round, idx)
                # torch.save(self.clients[idx].model.state_dict(),model_name)
            averaged_weights = self.defence_our(omega_locals, w_locals, w_local_pre)
        elif self.defence_mode in ['AE'] and self.attack_mode != "":
            # self.AE.model.load_state_dict('/home/haochu/Documents/Federated-Averaging-PyTorch/defense/AE_e10.pth')
            poisoning_idx = {}
            AE_input = list()
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                if self._round > 2 and 1<=it<=4:
                    for key in self.clients[idx].model.state_dict().keys():
                        local_weights[key] =  self.scale_attack * local_weights[key]
                    self.clients[idx].model.load_state_dict(local_weights)
                    plr = list(self.clients[idx].model.named_parameters())[-4][1].detach().numpy() * self.scale_attack
                else:
                    plr = list(self.clients[idx].model.named_parameters())[-4][1].detach().numpy()
                AE_input.append(plr)
            if 1 <= self._round <= 2:
                self.AE.train(AE_input)
            else:
                global_plr = list(self.model.named_parameters())[-4][1].detach().numpy()
                poisoning_idx = self.AE.test_latentspace(global_plr, AE_input, sampled_client_indices)
                # poisoning_idx = self.AE.test_IO(AE_input, sampled_client_indices)
                message = f"[!] Detected: {poisoning_idx}"
                print(message); logging.info(message)
                del message
            
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                if len(poisoning_idx) != 0:
                    if poisoning_idx[idx] == 1 and it < 5:
                        continue
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += coefficients[it] * local_weights[key]
        elif self.attack_mode != "":
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = coefficients[it] * local_weights[key]
                    elif it < 5:
                        averaged_weights[key] += coefficients[it] * (local_weights[key] + self.scale_attack * abs(averaged_weights[key] - local_weights[key]))
                    else:
                        averaged_weights[key] += coefficients[it] * local_weights[key]
        else:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += coefficients[it] * local_weights[key]
            
        new_global_weights = self.model.state_dict()
        for key in self.model.state_dict().keys():
            new_global_weights[key] = ((1-self.lr_server)*new_global_weights[key])
            new_global_weights[key] += self.lr_server*averaged_weights[key]
        
        self.model.load_state_dict(new_global_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # # evaluate selected clients with local dataset (same as the one used for local update)
        # if self.mp_flag:
        #     message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        #     print(message); logging.info(message)
        #     del message; gc.collect()

        #     with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
        #         workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        # else:
        #     self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        if 1 <= self._round <= 2 and self.defence_mode == 'AE':
            cnt = 0
            while cnt < 4:
                sampled_client_indices.pop(1)
                cnt += 1
            print('Clients: ', len(sampled_client_indices))
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)
        pred_labels, true_labels = np.array([]), np.array([])
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                pred_labels = np.concatenate([pred_labels, predicted.detach().cpu().numpy()], axis=None)
                true_labels = np.concatenate([true_labels, labels.detach().cpu().numpy()], axis=None)
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1score = f1_score(true_labels, pred_labels)
        return test_loss, test_accuracy, accuracy, precision, recall, f1score

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1score": [] }
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            test_loss, test_accuracy, accuracy, precision, recall, f1score = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(accuracy)
            self.results['precision'].append(precision)
            self.results['recall'].append(recall)
            self.results['f1score'].append(f1score)

            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Metrics', 
                {f"[{self.dataset_name}] Accuracy_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy,
                f"[{self.dataset_name}] Precision_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": precision,
                f"[{self.dataset_name}] Recall_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": recall,
                f"[{self.dataset_name}] F1-Score_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": f1score},
                self._round
                )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\
                \n\t=> Accuracy: {100. * accuracy:.2f}% - Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1score:.4f}\n"
                            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()
