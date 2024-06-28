from pix2pix.load_utils import initialize_training

# Change the path to your config file
# also be sure to update    ckpt_path: "/home/deanhazineh/Downloads/Pix2Pix/Demo/train_out/"
# in the config yaml
trainer = initialize_training("/home/deanhazineh/Downloads/Pix2Pix/Demo/config.yaml")
trainer.fit()
