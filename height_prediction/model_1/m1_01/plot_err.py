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

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_ylabel('Mean MSE')
#ax1.set_ylim([0, max(train_y)+0.01])

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Mean MSE')
#ax2.set_ylim([0, max(val_y)+0.01])

ax1.plot(train_x, train_y, c='b', label='Training data')
ax2.plot(val_x, val_y, c='r', label='Validation data')

leg = ax1.legend()
leg = ax2.legend()

plt.savefig('MSE_vs_epochs_m1.png')
plt.show()
