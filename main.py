import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset,DataLoader
from network import Pred
import torch.nn as nn

import matplotlib.pyplot as plt



def pre_process(df,train_size = 0.7):



    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    features = []
    labels = []

    for i in range(len(df)):
        features.append(scaled_data[i][:-1])
        labels.append(scaled_data[i][-1:])


    features = np.array(features)
    labels = np.array(labels)

    total_size = len(features)
    train_size = int(train_size * total_size)
    val_size = int(0.1 * total_size)

    train_x,val_x,test_x = features[:train_size],features[train_size:train_size+val_size],features[train_size+val_size:]
    train_y,val_y,test_y = labels[:train_size],labels[train_size:train_size+val_size],labels[train_size+val_size:]



    print("train_x shape is : {} , train_y shape is : {}".format(train_x.shape,train_y.shape))
    print("train_x shape is : {} , train_y shape is : {}".format(val_x.shape,val_y.shape))
    print("test_x shape is : {} , test_y shape is : {}".format(test_x.shape,test_y.shape))

    train_x = torch.tensor(train_x,dtype=torch.float32)
    train_y = torch.tensor(train_y,dtype=torch.float32)

    val_x = torch.tensor(val_x,dtype=torch.float32)
    val_y = torch.tensor(val_y,dtype=torch.float32)

    test_x = torch.tensor(test_x,dtype=torch.float32)
    test_y = torch.tensor(test_y,dtype=torch.float32)

    train_dataset = TensorDataset(train_x,train_y)
    val_dataset = TensorDataset(val_x,val_y)
    test_dataset = TensorDataset(test_x,test_y)
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=True)

    return train_loader,val_loader,test_loader


if __name__ == "__main__":
    df = pd.read_csv("output_clean_split.csv")
    train_loader,val_loader,test_loader = pre_process(df)

    model = Pred(67,128,1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    num_epochs = 20

    best_val_loss = float('inf')
    patience = 5
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x,batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output,batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)

        #validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                output= model(val_x)
                loss = criterion(output,val_y)
                val_loss+=loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict, "best_model.pt")
        else:
            counter +=1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    
    model.eval()
    predictions = []
    true_labels = []
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            predictions.append(output)
            true_labels.append(batch_y)
            loss = criterion(output,batch_y)
            test_loss+=loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    predictions = torch.cat(predictions).numpy()
    true_labels = torch.cat(true_labels).numpy()

    plt.plot(true_labels,linestyle="none",label="True",marker='o')
    plt.plot(predictions, linestyle='none',label="Predicted", marker='x')
    plt.title('Machine Learning Predictions vs. Original Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()