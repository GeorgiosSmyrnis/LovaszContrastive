# Lovasz Theta Contrastive Learning

### Authors: Georgios Smyrnis, Matt Jordan, Ananya Uppal, Giannis Daras, Alexandros G. Dimakis

This code is used to train the Lovasz Theta Contrastive models.

The code for CIFAR data is in the code_lovasz_contrastive, and contains the following:
- main_lovaszcon.py: The code implementing our method.
- main_linear.py: The code used to train the linear head on top of our model.
- main_ce.py: The code used for the crossentropy baseline.
- confusion_matrix_similarity.py: Derive the confusion matrix for the code.
- clip_similarity.py: The code to get similarities derived by CLIP embeddings
To train the model, run:
python main_lovaszcon.py --learning_rate 0.5 --temp 0.1 --batch_size 512 --epochs 300 --cosine --stable --sim_mat /path/to/similarity.csv
To train the linear head on top, run:
python main_linear.py --learning_rate 0.5 --batch_size 512 --epochs 10 --cosine --ckpt /path/to/contrastive/model.pth
This code is derived from the Supervised Contrastive Learning paper. The license for the original code is included.

The code for our ImageNet-100 experiments can be found in the code_moco_lovasz_folder, containing the following:
- main_moco.py: Performs training of our Lovasz contrastive loss with the MoCo trick.
- main_lincls.py: Trains the linear classifier on top of our method.
- moco/builder.py: Contains the loss functions.
- moco/loader.py: Contains other utilities.
Scripts to run our models can be found in the scripts/ and lin_scripts/ folders.
This code is derived from the Momentum Contrast paper. The license for the original code is included.

The code for the unsupervised experiments can be found in the folder code_lovasz_unsupervised, and can be run in a similar fashion to the supervised one.
