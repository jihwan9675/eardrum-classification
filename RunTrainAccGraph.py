import h5py
import os, sys
import numpy
import matplotlib.pyplot as plots

def main():
	history_path = 'history/BS32_AUG_KF10_eardrum_inceptionv3_013'
	if os.path.isdir(history_path):
		_IDs = next(os.walk(history_path))[2]
		if _IDs:
			ta = plots.figure('Train Accuracy').add_subplot(1,1,1)
			tl = plots.figure('Train Loss').add_subplot(1,1,1)
			va = plots.figure('Validation Accuracy').add_subplot(1,1,1)
			vl = plots.figure('Validation Loss').add_subplot(1,1,1)
			for _ID in _IDs:
				h5f = h5py.File(history_path+'/'+_ID, 'r')
				#print(history_path+'/'+_ID)
				for i in range (1):
					_train_accuracy = h5f['history'+str(i)+'_train_accuracy'][:]
					_train_loss = h5f['history'+str(i)+'_train_loss'][:]
					_val_accuracy = h5f['history'+str(i)+'_val_accuracy'][:]
					_val_loss = h5f['history'+str(i)+'_val_loss'][:]

					ta.plot(_train_accuracy)
					tl.plot(_train_loss)
					va.plot(_val_accuracy)
					vl.plot(_val_loss)

				ta.set_ylim((0, 1.1))
				tl.set_ylim(bottom=0)
				va.set_ylim((0, 1.1))
				vl.set_ylim(bottom=0)

				ta.grid(color='b', linestyle='dotted', linewidth=1)
				tl.grid(color='b', linestyle='dotted', linewidth=1)
				va.grid(color='b', linestyle='dotted', linewidth=1)
				vl.grid(color='b', linestyle='dotted', linewidth=1)

				ta.set_title('Train Accuracy')
				tl.set_title('Train Loss')
				va.set_title('Validation Accuracy')
				vl.set_title('Validation Loss')

				ta.set_xlabel('Epochs')
				tl.set_xlabel('Epochs')
				va.set_xlabel('Epochs')
				vl.set_xlabel('Epochs')

				ta.set_ylabel('Accuracy')
				tl.set_ylabel('Score')
				va.set_ylabel('Accuracy')
				vl.set_ylabel('Score')

				ta.legend(('Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9'), loc='lower right')
				tl.legend(('Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9'), loc='upper right')
				va.legend(('Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9'), loc='lower right')
				vl.legend(('Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9'), loc='upper right')

				h5f.close()

			plots.show()

		else:
			print("There is no record file.")
	else:
		print("There is no record folder.")

if __name__ == '__main__':
	try:
		main()

		sys.exit()
	except (EOFError, KeyboardInterrupt) as err:
		print("Keyboard Interrupted or Error!!!")