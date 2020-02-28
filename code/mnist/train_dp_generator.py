import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_ae import FCEnc, FCDec, ConvEnc, ConvDec
from models_gen import FCGen, FCLabelGen, FCCondGen
from aux import rff_gauss, get_mnist_dataloaders, plot_mnist_batch, meddistance, save_gen_labels, log_args, flat_data


def train(enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval, ae_conv, ae_label,
          do_gen_labels, uniform_labels):

  gen.train()
  for batch_idx, (data, labels) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))

    if not ae_conv:
      data = flat_data(data.to(device), labels.to(device), device, n_labels=10, add_label=ae_label)

    bs = labels.shape[0]

    if not do_gen_labels:
      loss = rff_mmd_loss(enc(data), gen(gen.get_code(bs, device)))

    elif uniform_labels:
      one_hots = pt.zeros(bs, 10, device=device)
      one_hots.scatter_(1, labels.to(device)[:, None], 1)
      gen_code, gen_labels = gen.get_code(bs, device)
      loss = rff_mmd_loss(enc(data), one_hots, gen(gen_code), gen_labels)

    else:
      one_hots = pt.zeros(bs, 10, device=device)
      one_hots.scatter_(1, labels.to(device)[:, None], 1)
      gen_enc, gen_labels = gen(gen.get_code(bs, device))
      loss = rff_mmd_loss(enc(data), one_hots, gen_enc, gen_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(enc, dec, gen, device, test_loader, rff_mmd_loss, epoch, batch_size, ae_conv, ae_label, ae_norm_data,
         do_gen_labels, uniform_labels, log_dir):
  gen.eval()
  gen_samples, gen_labels = None, None

  test_loss = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data = data.to(device)

      if not ae_conv:
        data = flat_data(data.to(device), labels.to(device), device, n_labels=10, add_label=ae_label)

      data_enc = enc(data)
      bs = labels.shape[0]

      if not do_gen_labels:
        gen_enc = gen(gen.get_code(bs, device))
        gen_labels = None
        loss = rff_mmd_loss(enc(data), gen_enc)

      elif uniform_labels:
        one_hots = pt.zeros(bs, 10, device=device)
        one_hots.scatter_(1, labels.to(device)[:, None], 1)
        gen_code, gen_labels = gen.get_code(bs, device)
        gen_enc = gen(gen_code)
        loss = rff_mmd_loss(enc(data), one_hots, gen_enc, gen_labels)

      else:
        one_hots = pt.zeros(bs, 10, device=device)
        one_hots.scatter_(1, labels.to(device)[:, None], 1)
        gen_enc, gen_labels = gen(gen.get_code(bs, device))
        loss = rff_mmd_loss(enc(data), one_hots, gen_enc, gen_labels)

      gen_samples = dec(gen_enc)

      test_loss += loss.item()  # sum up batch loss
    test_loss /= (len(test_loader.dataset) / batch_size)

    data_enc_batch = data_enc.cpu().numpy()
    med_dist = meddistance(data_enc_batch)
    print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

    if uniform_labels:
      ordered_labels = pt.repeat_interleave(pt.arange(10), 10)[:, None].to(device)
      gen_code, gen_labels = gen.get_code(100, device, labels=ordered_labels)
      gen_samples = dec(gen(gen_code))

    plot_samples = gen_samples[:100, ...].cpu().numpy()

    if ae_label:
      plot_samples = gen_samples[:, :784]
    plot_mnist_batch(plot_samples, 10, 10, log_dir + f'samples_ep{epoch}', denorm=ae_norm_data)

    if gen_labels is not None:
      save_gen_labels(gen_labels[:100, ...].cpu().numpy(), 10, 10, log_dir + f'labels_ep{epoch}')

  print('Test set: Average loss: {:.4f}'.format(test_loss))


def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, do_gen_labels, n_labels, noise_factor, batch_size):
  assert d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)

  if not do_gen_labels:
    def mean_embedding(x):
      return pt.mean(rff_gauss(x, w_freq), dim=0)

    def rff_mmd_loss(data_enc, gen_out):
      data_emb = mean_embedding(data_enc)
      gen_emb = mean_embedding(gen_out)
      noise = pt.randn(d_rff, device=device) * (2 * noise_factor / batch_size)
      return pt.sum((data_emb + noise - gen_emb) ** 2)
  else:
    def label_mean_embedding(data, labels):
      return pt.mean(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), labels]), 0)

    def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels):
      data_emb = label_mean_embedding(data_enc, labels)  # (d_rff, n_labels)
      gen_emb = label_mean_embedding(gen_enc, gen_labels)
      noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / batch_size)
      return pt.sum((data_emb + noise - gen_emb) ** 2)
  return rff_mmd_loss


def synthesize_mnist_with_uniform_labels(gen, dec, device, ae_label, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  assert n_data % gen_batch_size == 0
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():

    for idx in range(n_iterations):

      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = dec(gen(gen_code))

      if ae_label:
        gen_samples = gen_samples[:, :784]
      data_list.append(gen_samples)

  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--no-cuda', action='store_true', default=False, help='turns off gpu')
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--n-labels', type=int, default=10, help='number of labels/classes in data')
  parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--synth-mnist', action='store_true', default=True, help='if true, make 60k synthetic mnist')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=500)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=7)
  parser.add_argument('--lr', '-lr', type=float, default=1e-3, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
  parser.add_argument('--gen-labels', action='store_true', default=True, help='generate labels as well as samples')
  parser.add_argument('--uniform-labels', action='store_true', default=True, help='assume uniform label distribution')
  parser.add_argument('--d-enc', '-denc', type=int, default=10, help='embedding space dimensionality')
  parser.add_argument('--d-code', '-dcode', type=int, default=10, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='100,100', help='specifies hidden layers of generator')

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=10000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=float, default=80.0, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=0.7, help='privacy noise parameter')

  # Autoencoder info
  parser.add_argument('--ae-no-bias', action='store_true', default=True, help='if true, AE has no bias units')
  parser.add_argument('--ae-bn', action='store_true', default=True, help='if true, AE uses batch norm')
  parser.add_argument('--ae-enc-spec', type=str, default='300,100', help='AE encoder hidden layers')
  parser.add_argument('--ae-dec-spec', type=str, default='100', help='AE decoder hidden layers')
  parser.add_argument('--ae-norm-data', action='store_true', default=False, help='if true, data was pre-processed')
  parser.add_argument('--ae-load-dir', type=str, default=None, help='path to where AE is stored')

  # outdated variants and args only relevant for automated loading - don't worry about those
  parser.add_argument('--ae-conv', action='store_true', default=False)
  parser.add_argument('--ae-label', action='store_true', default=False)
  parser.add_argument('--ae-ce-loss', action='store_true', default=False)
  parser.add_argument('--ae-clip', type=float, default=None)
  parser.add_argument('--ae-noise', type=float, default=None)
  parser.add_argument('--ae-siam-weight', '-wsiam', type=float, default=None)
  parser.add_argument('--ae-siam-margin', '-msiam', type=float, default=None)
  ar = parser.parse_args()

  if ar.log_dir is None:
    ar.log_dir = get_log_dir(ar)
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)
  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar


def get_log_dir(ar):
  if ar.log_name is not None:
    log_dir = ar.base_log_dir + ar.log_name + '/'
  else:
    gen_type = f'{"uniform_" if ar.uniform_labels else ""}{"labeled_" if ar.gen_labels else "unlabeled_"}'
    gen_spec = f'd{ar.d_enc}_gen{ar.gen_spec}_sig{ar.noise_factor}_dcode{ar.d_code}_drff{ar.d_rff}_rffsig{ar.rff_sigma}'
    ae_type = f'{"label_" if ar.ae_label else ""}{"ce_" if ar.ae_ce_loss else "mse_"}'
    ae_spec = f'enc{ar.ae_enc_spec}_dec{ar.ae_dec_spec}_clip{ar.ae_clip}_sig{ar.ae_noise}'
    ae_siam = f'_siam_w{ar.ae_siam_weight}_m{ar.ae_siam_margin}' if ar.ae_siam_weight > 0. else ''
    log_dir = ar.base_log_dir + gen_type + gen_spec + '_AE:' + ae_type + ae_spec + ae_siam + '/'
  return log_dir


def preprocess_args(args):
  if args.seed is None:
    args.seed = np.random.randint(0, 1000)

  assert args.gen_labels or not args.uniform_labels
  assert args.ae_load_dir is not None

def main():
  # Training settings

  ar = get_args()

  pt.manual_seed(ar.seed)

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda,
                                                    normalize=ar.ae_norm_data, dataset=ar.data)

  device = pt.device("cuda" if use_cuda else "cpu")

  d_data = 28**2+ar.n_labels if ar.ae_label else 28**2

  enc_spec = tuple([int(k) for k in ar.ae_enc_spec.split(',')])
  dec_spec = tuple([int(k) for k in ar.ae_dec_spec.split(',')])

  if ar.ae_conv:
    enc = ConvEnc(ar.d_enc, enc_spec, extra_conv=True)
    dec = ConvDec(ar.d_enc, dec_spec, use_sigmoid=not ar.ae_norm_data)
  else:
    enc = FCEnc(d_in=d_data, d_hid=enc_spec, d_enc=ar.d_enc, batch_norm=ar.ae_bn)
    dec = FCDec(d_enc=ar.d_enc, d_hid=dec_spec, d_out=d_data,
                use_sigmoid=not ar.ae_norm_data, use_bias=not ar.ae_no_bias)

  enc.load_state_dict(pt.load(ar.ae_load_dir + 'enc.pt'))
  dec_extended_dict = pt.load(ar.ae_load_dir + 'dec.pt')
  dec_reduced_dict = {k: dec_extended_dict[k] for k in dec.state_dict().keys()}
  dec.load_state_dict(dec_reduced_dict)

  enc = enc.to(device)
  dec = dec.to(device)

  gen_spec = tuple([int(k) for k in ar.gen_spec.split(',')]) if ar.gen_spec is not None else None
  if ar.gen_labels:
    if ar.uniform_labels:
      gen = FCCondGen(ar.d_code, gen_spec, ar.d_enc, ar.n_labels, batch_norm=ar.batch_norm)
    else:
      gen = FCLabelGen(ar.d_code, gen_spec, ar.d_enc, ar.n_labels, batch_norm=ar.batch_norm)
  else:
    gen = FCGen(ar.d_code, gen_spec, ar.d_enc, batch_norm=ar.batch_norm)
  gen = gen.to(device)

  rff_mmd_loss = get_rff_mmd_loss(ar.d_enc, ar.d_rff, ar.rff_sigma, device, ar.gen_labels,
                                  ar.n_labels, ar.noise_factor, ar.batch_size)

  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
  for epoch in range(1, ar.epochs + 1):
    train(enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss, ar.log_interval,
          ar.ae_conv, ar.ae_label, ar.gen_labels, ar.uniform_labels)
    test(enc, dec, gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size,
         ar.ae_conv, ar.ae_label, ar.ae_norm_data, ar.gen_labels, ar.uniform_labels, ar.log_dir)
    scheduler.step()

  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')
  if ar.synth_mnist:
    assert ar.uniform_labels
    syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, dec, device, ar.ae_label)
    np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)


if __name__ == '__main__':
  main()
