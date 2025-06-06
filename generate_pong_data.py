import numpy as np
import tensorflow as tf
import os

def label_move(ball_x, ball_y, ball_vel_x, ball_vel_y, paddle_y):
    # If the ball is above the current paddle location, move up, and vice versa. Else, do not move.
    if ball_y < paddle_y - 0.05:
        return -1.0
    elif ball_y > paddle_y + 0.05:
        return 1.0
    else:
        return 0.0

# Run 50,000 samples and create arrays for data (ball location/speed information) and labels (which direction
# paddle is moved in response)
NUM_SAMPLES = 50000
data   = []
labels = []

for _ in range(NUM_SAMPLES):

    # Spawn ball at random location
    bx  = np.random.rand()
    by  = np.random.rand()

    # Calculate the random number from [0, 1.0] -> [-0.5, 0.5] for an even change of being negative or positive
    # -> [-0.01, 0.01] for a smaller range since Pong does not move the ball as quickly as other games. 
    # Gives velocity
    bvx = (np.random.rand() - 0.5) * 0.02
    bvy = (np.random.rand() - 0.5) * 0.02

    # Holds paddle location so AI can react wherever
    py  = np.random.rand()

    mv = label_move(bx, by, bvx, bvy, py)
    data.append([bx, by, bvx, bvy, py])
    labels.append([mv])

# Convert lists to NumPy arrays.
X = np.array(data,   dtype=np.float32)
y = np.array(labels, dtype=np.float32)

# Using sequential model since we just need 1 input for 1 output
model = tf.keras.Sequential([
    # Using 32 is a design choice and this number can be changed. Since the data array is five dimensional.
    # Relu is used because of the binary options of no (negative) or yes (how much, positive).
    # "Entry layer", "Processing layer", "Output layer"
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 7) Train for 10 epochs, batch size 256, 10% validation split
model.fit(
    X, y,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=2
)

# 8) Save the model to "saved_pong_ai"
SAVE_DIR = "saved_pong_ai"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
model.save(SAVE_DIR)

print(f"Model saved into: ./{SAVE_DIR}")
