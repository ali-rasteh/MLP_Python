from functions import learn, ds_read
from mlp import MLP

# 7 datasets

# Dataset 1 : Banana
print("Learning the Banana dataset")
Banana_ds_x, Banana_ds_y = ds_read('Banana')

Banana_net = MLP(2, 10, 10, 1)
Banana_net.reset()
learn(Banana_net, Banana_ds_x, Banana_ds_y, True, epochs=1, lrate=0.001, momentum=0.001)

#
# Dataset 2 : Haberman
print("\n \nLearning the Haberman dataset")
Haberman_ds_x, Haberman_ds_y = ds_read('Haberman')

Haberman_net = MLP(3, 3, 1)
Haberman_net.reset()
learn(Haberman_net, Haberman_ds_x, Haberman_ds_y, True, epochs=1, lrate=0.1, momentum=0.1)

#
# Dataset 3 : Titanic
print("\n \nLearning the Titanic dataset")
Titanic_ds_x, Titanic_ds_y = ds_read('Titanic')

Titanic_net = MLP(3, 12, 12, 1)
Titanic_net.reset()
learn(Titanic_net, Titanic_ds_x, Titanic_ds_y, True, epochs=1, lrate=0.01, momentum=0.1)

#
# Dataset 4 : Balance
print("\n \nLearning the Balance dataset")
Balance_ds_x, Balance_ds_y = ds_read('Balance')

Balance_net = MLP(4, 12, 12, 3)
Balance_net.reset()
learn(Balance_net, Balance_ds_x, Balance_ds_y, False, epochs=1, lrate=0.001, momentum=0.1)

#
# Dataset 5 : Hayes-roth
print("\n \nLearning the Hayes-roth dataset")
Hayes_roth_ds_x, Hayes_roth_ds_y = ds_read('Hayes-roth')

Hayes_roth_net = MLP(4, 5, 3)
Hayes_roth_net.reset()
learn(Hayes_roth_net, Hayes_roth_ds_x, Hayes_roth_ds_y, False, epochs=1, lrate=0.1, momentum=0.1)

#
# Dataset 6 : Newthyroid
print("\n \nLearning the Newthyroid dataset")
Newthyroid_ds_x, Newthyroid_ds_y = ds_read('Newthyroid')

Newthyroid_net = MLP(5, 5, 3)
Newthyroid_net.reset()
learn(Newthyroid_net, Newthyroid_ds_x, Newthyroid_ds_y, False, epochs=1, lrate=0.1, momentum=0.1)

#
# Dataset 7 : Wine
print("\n \nLearning the Wine dataset")
Wine_ds_x, Wine_ds_y = ds_read('Wine')

Wine_net = MLP(13, 11, 6, 3)
Wine_net.reset()
learn(Wine_net, Wine_ds_x, Wine_ds_y, False, epochs=1, lrate=0.1, momentum=0.1)
