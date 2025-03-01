class args():
	# training args
	epochs = 60  # "number of training epochs, default is 2"
	batch_size = 8  # 8 #"batch size for training, default is 4"
	dataset_ir = "../LLVIP/train/infrared/"
	dataset_vi = "../LLVIP/train/visible/"
	# dataset_ir = "../M3FD/crop_ir/"
	# dataset_vi = "../M3FD/crop_vi/"
	HEIGHT = 512
	WIDTH = 640

	train_num = 10000
	test_num = 40

	test_path = "../images/LLVIP/visible/"
	output_path = '../outputs/'

	input_nc = 1
	output_nc = 1
	img_flag = 'L'  # 'RGB'#

	save_model_dir = "models"  # "path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	# image_height = 1024  # "size of training images, default is 512 × 640"
	# image_width = 1280
	cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"
	seed = 1  # "random seed for training"
	# ssim_weight = [1, 10, 100, 1000, 10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  # "learning rate, default is 0.0001"
	lr_light = 1e-4  # "learning rate, default is 0.0001"
	log_interval = 10  # "number of images after which the training loss is logged, default is 500"
	resume = None  # "./models/BTSFusion.model"
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	model_path = "models/test_model.pth"

