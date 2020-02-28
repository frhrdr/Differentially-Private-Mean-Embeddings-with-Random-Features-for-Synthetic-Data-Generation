# This is for computing the cumulative privacy loss of our algorithm
# We use the analytic moments accountant method by Wang et al
# (their github repo is : https://github.com/yuxiangw/autodp)
# by changing the form of upper bound on the Renyi DP, resulting from
# several Gaussian mechanisms we use given a mini-batch.
from autodp import rdp_acct, rdp_bank


def main():

    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma = 0.96  # 0.7 for DP-MERF+AE, 0.595, 0.96 & 1.8 for DP-MERF

    # (2) desired delta level
    delta = 1e-5

    # (5) number of training steps
    n_epochs = 5  # 5 for DP-MERF and 17 for DP-MERF+AE
    batch_size = 500  # the same across experiments

    n_data = 60000  # fixed for mnist
    steps_per_epoch = n_data // batch_size

    n_steps = steps_per_epoch * n_epochs

    # (6) sampling rate
    prob = batch_size / n_data

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    eps_seq = []
    for i in range(1, n_steps+1):
        acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), prob)
        if i % steps_per_epoch == 0 or i == n_steps:
            eps_seq.append(acct.get_eps(delta))
            print("[", i, "]Privacy loss is", (eps_seq[-1]))


if __name__ == '__main__':
    main()
