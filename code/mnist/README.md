# Experiments on MNIST

## Repository structure

- `train_dp_generator_directly.py` trains a DP-MERF generator and can create a synthetic dataset afterwards.
- `train_dp_autoencoder.py` trains the autoencoder for a DP-MERF+AE model.
- `train_dp_generator.py` trains the generator for a DP-MERF+AE model and can create a synthetic dataset afterwards.
- `synth_data_benchmark.py` evaluates synthetic datasets on classification tasks.
- `dp_analysis.py` computes privacy loss for training DP-MERF models.
- `dp_cgan_reference.py` trains CGAN generator and generates synthetic dataset.
- remaining files contain helper functions and are not meant to be executed.

## Running the code

In order to reproduce our experiments, you can run the commands outlined below.
All hyperparameters have been set to the values used in the paper and can be examined in the respective files.
Please note, that DP-MERF downloads datasets, while the DP-CGAN code assumes they already exist, so make sure to run DP-MERF first.

After running the code indicated below, synthetic datasets for each experiment can be found in the `logs/gen/exeriment_name/` directories for further use, along with plots of generated samples after each training epoch.
Results of the `synth_data_benchmark.py` test are printed to command line by default but can also be stored in the same directory by adding the `--log-results` flag.

#### privacy analysis

All privacy settings in the scripts below are set for (9.6, 10^-5)-DP by default. Parameters for different privacy settings can be computed by running 
`python3 dp_analysis.py`, after changing the parameters defined in that script.

## digit MNIST

#### DP-MERF
For the (2.9,10^-5)-DP model, append -noise 0.96 and for (1.3,10^-5)-DP, append -noise 1.8  
- `python3 train_dp_generator_directly.py --log-name dp_merf_digits --data digits`

#### DP-MERF+AE
first autoencoder training, then generator training
- `python3 train_dp_autoencoder.py --log-name dp_merf_ae_digits --data digits`
- `python3 train_dp_generator.py --ae-load-dir logs/ae/dp_merf_ae_digits/ --log-name dp_merf_ae_digits --data digits`

#### DP-CGAN
- `python3 dp_cgan_reference.py --data-save-str dpcgan_digits --data digits`

#### Evaluation
each of the above models creates a synthetic dataset, which can be evaluated by running the following script with the previously used experiment name (`log-name` or `data-save-str`)
- `python3 synth_data_benchmark.py --data-log-name *experiment name* --data digits`

## fashion MNIST

All experiments are run with the same hyperparameters. The only change requires is switching the `--data` flag to `fashion`

#### DP-MERF
- `python3 train_dp_generator_directly.py --log-name dp_merf_fashion --data fashion`
- `python3 synth_data_benchmark.py --data-log-name dp_merf_fashion --data fashion`

#### DP-MERF+AE
- `python3 train_dp_autoencoder.py --log-name dp_merf_ae_fashion --data fashion`
- `python3 train_dp_generator.py --ae-load-dir logs/ae/dp_merf_ae_fashion/ --log-name dp_merf_ae_fashion --data fashion`
- `python3 synth_data_benchmark.py --data-log-name dp_merf_ae_fashion --data fashion`

#### DP-CGAN
- `python3 dp_cgan_reference.py --data-save-str dpcgan_fashion --data fashion`
- `python3 synth_data_benchmark.py --data-log-name dpcgan_fashion --data fashion`