import sys
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    content = f.readlines()

content = [x.strip() for x in content]

steps = []
loss = []

for i in content:
	words = i.split()
	if len(words) > 2:
		steps.append(words[0])
		loss.append(words[1])

#print "steps ", steps
#print "loss ", loss

plt.plot(steps,loss)
plt.show()