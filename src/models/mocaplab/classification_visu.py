
import torch
from torch.utils.data import DataLoader

#sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")


from cnn.cnn_dataset import MocaplabDatasetCNN
from cnn.cnn import TestCNN
from fc.fc_dataset import MocaplabDatasetFC
from fc.fc import MocaplabFC
from lstm.lstm_dataset import MocaplabDatasetLSTM
from lstm.lstm import LSTM

if __name__ == "__main__":

    print('---------- CNN ----------')

    # Load the data
    print('### Data Loading ###')
    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetCNN(data_path, padding=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model
    model = TestCNN()
    model.load_state_dict(torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/CNN_20240514_211739.ckpt"))

    f = open("self_supervised_learning/dev/ProjetCassiopee/train_results/cnn.txt", "w")

    for batch in data_loader :

        data, label, name = batch
        #data = data.to(device)
        #label = label.to(device)
    
        output = model(data)

        _, predicted = torch.max(output.data, dim=1)

        if predicted != label :
            print(name[0], ":\tPredicted : ", predicted.tolist()[0], "/ Label : ", label.tolist()[0])
            f.write("\n/!\\ " + name[0] + ":\tPredicted : " + str(predicted.tolist()[0]) + " / Label : " + str(label.tolist()[0]) + "\n\n")
        else :
            f.write(name[0] + ":\tPredicted : " + str(predicted.tolist()[0]) + " / Label : " + str(label.tolist()[0]) + "\n")
    
    f.close()




    print('---------- FC ----------')

    # Load the data
    print('### Data Loading ###')
    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetFC(data_path, padding=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model
    model = MocaplabFC()
    model.load_state_dict(torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/FC_20240514_215113.ckpt"))

    f = open("self_supervised_learning/dev/ProjetCassiopee/train_results/fc.txt", "w")

    for batch in data_loader :

        data, label, name = batch
        #data = data.to(device)
        #label = label.to(device)

        data = data.view(data.size(0), -1)
        data = data.to(torch.float32)
        output = model(data)

        _, predicted = torch.max(output.data, dim=1)

        if predicted != label :
            print(name[0], ":\tPredicted : ", predicted.tolist()[0], "/ Label : ", label.tolist()[0])
            f.write("\n/!\\ " + name[0] + ":\tPredicted : " + str(predicted.tolist()[0]) + " / Label : " + str(label.tolist()[0]) + "\n\n")
        else :
            f.write(name[0] + ":\tPredicted : " + str(predicted.tolist()[0]) + " / Label : " + str(label.tolist()[0]) + "\n")

    f.close()



    print('---------- LSTM ----------')

    # Load the data
    print('### Data Loading ###')
    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetLSTM(data_path, return_filename=True, padding=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model
    model = LSTM(input_size=237, hidden_size=16, num_layers=2, output_size=2)
    model.load_state_dict(torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/LSTM_20240514_180721.ckpt"))

    f = open("self_supervised_learning/dev/ProjetCassiopee/train_results/lstm.txt", "w")

    for batch in data_loader :
            
        data, label, name = batch
        #data = data.to(device)
        #label = label.to(device)
        data = data.to(torch.float32)
        output = model(data)

        _, predicted = torch.max(output.data, dim=1)

        if predicted != label :
            print(name[0], ":\tPredicted : ", predicted.tolist()[0], "/ Label : ", label.tolist()[0])
            f.write("\n/!\\ " + name[0] + ":\tPredicted : " + str(predicted.tolist()[0]) + " / Label : " + str(label.tolist()[0]) + "\n\n")
        else :
            f.write(name[0] + ":\tPredicted : " + str(predicted.tolist()[0]) + " / Label : " + str(label.tolist()[0]) + "\n")

    f.close()

    print('### End of the script ###')
