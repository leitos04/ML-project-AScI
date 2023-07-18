import matplotlib.pyplot as plt

#------------------------- Mean MSE vs epochs -------------------------

with open("results/mse_training_values.txt") as file:
    train_data = file.read()
    train_data = train_data.split('\n')
    train_data = [row for row in train_data if row]  # Remove empty lines
    train_x = [int(row.split()[0]) for row in train_data]
    #print(train_x)
    train_y = [float(row.split()[1]) for row in train_data]
    #print(train_y)
    
with open("results/mse_validation_values.txt") as file:
    val_data = file.read()
    val_data = val_data.split('\n')
    val_data = [row for row in val_data if row]  # Remove empty lines
    val_x = [int(row.split()[0]) for row in val_data]
    #print(val_x)
    val_y = [float(row.split()[1]) for row in val_data]
    #print(val_y)

plt.title('Mean MSE vs Epochs')
plt.plot(train_x, train_y, c='b', label='Training data')
plt.plot(val_x, val_y, c='r', label='Validation data')
plt.xlabel('Epochs')
plt.ylabel('Mean MSE')

leg = plt.legend()

plt.savefig('results/Mean_MSE_vs_epochs.png')
plt.show()

#------------------------- Mean MAE vs epochs -------------------------

with open("results/mae_training_values.txt") as file:
    train_data = file.read()
    train_data = train_data.split('\n')
    train_data = [row for row in train_data if row]  # Remove empty lines
    mae_train_x = [int(row.split()[0]) for row in train_data]
    #print(train_x)
    mae_train_y = [float(row.split()[1]) for row in train_data]
    #print(train_y)

with open("results/mae_validation_values.txt") as file:
    val_data = file.read()
    val_data = val_data.split('\n')
    val_data = [row for row in val_data if row]  # Remove empty lines
    mae_val_x = [int(row.split()[0]) for row in val_data]
    #print(val_x)
    mae_val_y = [float(row.split()[1]) for row in val_data]
    #print(val_y)

plt.title('Mean MAE vs Epochs')
plt.plot(mae_train_x, mae_train_y, c='b', label='Training data')
plt.plot(mae_val_x, mae_val_y, c='r', label='Validation data')
plt.xlabel('Epochs')
plt.ylabel('Mean MAE')

leg = plt.legend()

plt.savefig('results/Mean_MAE_vs_epochs.png')
plt.show()




