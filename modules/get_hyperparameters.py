import configparser as CP

class GetHyperparameters():
	# Attach Hyperparameters File
	HParameters = CP.RawConfigParser()
	HParameters.readfp(open(r'./Hyperparameters.txt'))

	# get Hyperparameters
	batch_size16 = int(HParameters.get('train settings', 'batch_size16'))
	batch_size32 = int(HParameters.get('train settings', 'batch_size32'))
	epochs = int(HParameters.get('train settings', 'epochs'))

	image_path = HParameters.get('data paths', 'image_path')

	def train_parameters(self):
		return self.batch_size16, self.batch_size32, self.epochs

	def get_image_path(self):
		return self.image_path