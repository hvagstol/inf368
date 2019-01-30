import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np


sns.set()
sns.set_style("dark")


# summarize history for validation accuracy
plt.clf()

all_val_acc = []
for index in range(5):
	lenet_history = None
	with open('output/lenet-cv-'+str(index)+'_history.json', 'r') as f:
		lenet_history = json.loads(f.read())

	plt.plot(lenet_history['val_acc'])
	all_val_acc.append(lenet_history['val_acc'][-1])

plt.title('lenet stratified k fold cross validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['lenet-cv-0', 'lenet-cv-1', 'lenet-cv-2', 'lenet-cv-3', 'lenet-cv-4'], loc='upper left')
plt.savefig('output/lenet_cv_validation_accuracy.png', bbox_inches='tight')

#print(all_val_acc)
print('Validation accuracy mean %.6f' % np.mean(all_val_acc))
print('Validation accuracy variance %.6f' % np.var(all_val_acc))

with open('output/lenet_cv_val_loss.txt','w') as f:
    f.write('Validation accuracy\nMean: '+str(np.mean(all_val_acc))+' Variance:'+ str(np.var(all_val_acc)))
