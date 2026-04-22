import matplotlib.pyplot as plt

steps = []
losses = []

with open("losslogs.txt", "r") as f:
    next(f)  
    for line in f:
        step, epoch, loss, eta = line.strip().split(",")
        steps.append(int(step))
        losses.append(float(loss))

plt.figure()
plt.plot(steps, losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss vs Steps")
plt.grid()
plt.show()