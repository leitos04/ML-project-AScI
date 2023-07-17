import matplotlib.pyplot as plt


with open("loss_training_values.txt") as file:
    train_data = file.read()
    train_data = train_data.split('\n')
    train_data = [row for row in train_data if row]  # Remove empty lines
    #print(train_data)
    train_x = [int(row.split()[0]) for row in train_data]
    print(train_x)
    train_y = [float(row.split()[1]) for row in train_data]
    print(train_y)
    
with open("loss_validation_values.txt") as file:
    val_data = file.read()
    val_data = val_data.split('\n')
    val_data = [row for row in val_data if row]  # Remove empty lines
    #print(val_data)
    val_x = [int(row.split()[0]) for row in val_data]
    print(val_x)
    val_y = [float(row.split()[1]) for row in val_data]
    print(val_y)

plt.plot(train_x, train_y, c='b', label='Training data')
plt.plot(val_x, val_y, c='r', label='Validation data')
plt.xlabel('Epochs')
plt.ylabel('Mean MSE')

leg = plt.legend()

plt.savefig('MSE_vs_epochs_m1.png')
plt.show()
