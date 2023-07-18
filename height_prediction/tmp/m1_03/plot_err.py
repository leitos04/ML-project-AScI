import matplotlib.pyplot as plt


with open("training_loss_values.txt") as file:
    train_data = file.read()
    train_data = train_data.split('\n')
    train_data = [row for row in train_data if row]  # Remove empty lines
    #print(train_data)
    train_x = [int(row.split()[0]) for row in train_data]
    print(train_x)
    train_y = [float(row.split()[1]) for row in train_data]
    print(train_y)
    
with open("validation_loss_values.txt") as file:
    val_data = file.read()
    val_data = val_data.split('\n')
    val_data = [row for row in val_data if row]  # Remove empty lines
    #print(val_data)
    val_x = [int(row.split()[0]) for row in val_data]
    print(val_x)
    val_y = [float(row.split()[1]) for row in val_data]
    print(val_y)

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

#ax1.set_title("Training data")    
#ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean MSE')

#ax2.set_title("Validation data")    
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Mean MSE')

ax1.plot(train_x, train_y, c='b', label='Training data')
ax2.plot(val_x, val_y, c='r', label='Validation data')

leg = ax1.legend()
leg = ax2.legend()

plt.show()
