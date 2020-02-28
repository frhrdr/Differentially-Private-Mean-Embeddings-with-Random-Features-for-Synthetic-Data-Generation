import torch.nn as nn


class FCEnc(nn.Module):
  def __init__(self, d_in, d_hid, d_enc, batch_norm=False):
    super(FCEnc, self).__init__()
    # nc = (1, 4, 8, 16)  # n channels
    if d_hid is None:
      self.fc1 = nn.Linear(d_in, d_enc)
      self.fc2 = None
      self.fc3 = None
      self.bn1 = None
      self.bn2 = None
    elif len(d_hid) == 1:
      self.fc1 = nn.Linear(d_in, d_hid[0])
      self.fc2 = nn.Linear(d_hid[0], d_enc)
      self.fc3 = None
      self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
      self.bn2 = None
    elif len(d_hid) == 2:
      self.fc1 = nn.Linear(d_in, d_hid[0])
      self.fc2 = nn.Linear(d_hid[0], d_hid[1])
      self.fc3 = nn.Linear(d_hid[1], d_enc)
      self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
      self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
    else:
      raise ValueError

    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.bn1 is not None else x
    x = self.fc2(self.relu(x)) if self.fc2 is not None else x
    x = self.bn2(x) if self.bn2 is not None else x
    x = self.fc3(self.relu(x)) if self.fc3 is not None else x
    return x


class FCDec(nn.Module):
  def __init__(self, d_enc, d_hid, d_out, use_sigmoid=False, use_bias=True):
    super(FCDec, self).__init__()
    if d_hid is None:
      self.fc1 = nn.Linear(d_enc, d_out, bias=use_bias)
      self.fc2 = None
      self.fc3 = None
    elif len(d_hid) == 1:
      self.fc1 = nn.Linear(d_enc, d_hid[0], bias=use_bias)
      self.fc2 = nn.Linear(d_hid[0], d_out, bias=use_bias)
      self.fc3 = None
    elif len(d_hid) == 2:
      self.fc1 = nn.Linear(d_enc, d_hid[0], bias=use_bias)
      self.fc2 = nn.Linear(d_hid[0], d_hid[1], bias=use_bias)
      self.fc3 = nn.Linear(d_hid[1], d_out, bias=use_bias)
    else:
      raise ValueError

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(self.relu(x)) if self.fc2 is not None else x
    x = self.fc3(self.relu(x)) if self.fc3 is not None else x
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x


# class FCEnc(nn.Module):
#
#   def __init__(self, d_in, d_hid, d_enc, reshape=False):
#     super(FCEnc, self).__init__()
#     # nc = (1, 4, 8, 16)  # n channels
#     layer_spec = [d_in] + list(d_hid) + [d_enc]
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k+1]) for k in range(len(layer_spec) - 1)]
#     self.relu = nn.ReLU()
#     self.reshape = reshape
#
#   def forward(self, x):
#     if self.reshape:
#       x = x.reshape(x.shape[0], -1)
#
#     for layer in self.fc_layers[:-1]:
#       x = self.relu(layer(x))
#     x = self.fc_layers[-1](x)
#
#     return x
#
#
# class FCDec(nn.Module):
#
#   def __init__(self, d_enc, d_hid, d_out, use_sigmoid=False, reshape=False):
#     super(FCDec, self).__init__()
#     layer_spec = [d_enc] + list(d_hid) + [d_out]
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.relu = nn.ReLU()
#     self.sigmoid = nn.Sigmoid()
#     self.use_sigmoid = use_sigmoid
#     self.reshape = reshape
#
#   def forward(self, x):
#     for layer in self.fc_layers[:-1]:
#       x = self.relu(layer(x))
#     x = self.fc_layers[-1](x)
#     if self.use_sigmoid:
#       x = self.sigmoid(x)
#     if self.reshape:
#       x = x.reshape(x.shape[0], 1, 28, 28)
#     return x


class ConvEnc(nn.Module):

  def __init__(self, d_enc, nc=(1, 2, 4, 4), extra_conv=False):
    super(ConvEnc, self).__init__()
    # nc = (1, 4, 8, 16)  # n channels
    self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(nc[1], nc[2], kernel_size=4, stride=2, padding=1)  # down to 14x14
    self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
    self.conv4 = nn.Conv2d(nc[2], nc[3], kernel_size=4, stride=2, padding=1)  # down to 7x7
    self.fc = nn.Linear(7*7*nc[3], d_enc)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x)) if self.conv3 is not None else x
    x = self.relu(self.conv4(x))
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x


class ConvDec(nn.Module):

  def __init__(self, d_enc, nc=(4, 4, 2, 1), use_sigmoid=False, use_bias=True):
    super(ConvDec, self).__init__()
    self.fc = nn.Linear(d_enc, 7*7*nc[0], bias=use_bias)
    self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1, bias=use_bias)
    self.conv2 = nn.Conv2d(nc[1], nc[2], kernel_size=3, stride=1, padding=1, bias=use_bias)  # up to 14x14
    self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1, bias=use_bias)
    self.conv4 = nn.Conv2d(nc[2], nc[3], kernel_size=3, stride=1, padding=1, bias=use_bias)  # up to 28x28
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.nc = nc
    self.use_sigmoid = use_sigmoid
    # N Params:
    # 2, 4, 2, 1 -> 5x49x2 + 4x4x9 + 4x2x9 + 2x2x9 + 2x1x9 + =  10*49 + (16 + 8 + 4 + 2)*9  = 490 + 270 = 780
    # 4, 4, 2, 1 -> 5x49x4 + 4x4x9 + 4x2x9 + 2x2x9 + 2x1x9 + =  20*49 + (16 + 8 + 4 + 2)*9  = 980 + 270 = 1250
    # 8, 8, 4, 1 -> 5x49x8 + 8x8x9 + 8x4x9 + 4x4x9 + 4x1x9 = 40*49 + (64 + 32 + 16 + 4)*9 = 1960 + 1044 = 4004
    # 16, 16, 8, 1 -> 5x49x16 + 16x16x9 + 16x8x9 + 8x8x9 + 8x1x9 = 80*49 + (256 + 128 + 64 + 4)*9 = 3820 + 4068 = 7888

  def forward(self, x):
    x = self.relu(self.fc(x))
    x = x.reshape(x.shape[0], self.nc[0], 7, 7)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x)) if self.conv3 is not None else x
    x = self.upsamp(x)
    x = self.conv4(x)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x


class ConvDecThin(nn.Module):

  def __init__(self, d_enc, nc=(4, 4, 2, 1), use_sigmoid=False):
    super(ConvDecThin, self).__init__()
    self.fc = nn.Linear(d_enc, 4*4*nc[0])
    self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(nc[1], nc[1], kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(nc[1], nc[2], kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1)
    self.conv5 = nn.Conv2d(nc[2], nc[3], kernel_size=3, stride=1)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.nc = nc
    self.use_sigmoid = use_sigmoid
    # N Params:
    # 2, 4, 2, 1 -> 5x16x2 +  2x4x9 + 4x4x9 + 4x2x9 + 2x2x1 + 2x1x9 + =  20*16 + (16 + 2*8 + 4 + 2)*9  = 160 + 342 = 502
    # 4, 4, 2, 1 -> 5x16x4 + 2 * 4x4x9 + 4x2x9 + 2x2x1 + 2x1x9 + =  20*16 + (2*16 + 8 + 4 + 2)*9  = 320 + 414 = 734

  def forward(self, x):
    x = self.relu(self.fc(x))
    x = x.reshape(x.shape[0], self.nc[0], 4, 4)
    x = self.relu(self.conv1(x))  # 4x4
    x = self.upsamp(x)  # 8x8
    x = self.relu(self.conv2(x))  # 8x8
    x = self.upsamp(x)  # 16x16
    x = self.relu(self.conv3(x))  # 16x16
    x = self.upsamp(x)  # 32x32
    x = self.conv4(x)  # 30x30
    x = self.conv5(x)  # 28x28
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x


class ConvDecFlat(nn.Module):

  def __init__(self, d_enc, nc=(4, 4, 1), use_sigmoid=False, use_bias=True):
    super(ConvDecFlat, self).__init__()
    self.fc = nn.Linear(d_enc, 7*7*nc[0], bias=use_bias)
    self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1, bias=use_bias)
    self.conv2 = nn.Conv2d(nc[1], nc[2], kernel_size=3, stride=1, padding=1, bias=use_bias)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.nc = nc
    self.use_sigmoid = use_sigmoid
    # N Params:
    # 2, 4, 2, 1 -> 5x49x2 + 4x4x9 + 4x2x9 + 2x2x9 + 2x1x9 + =  10*49 + (16 + 8 + 4 + 2)*9  = 490 + 270 = 780
    # 4, 4, 2, 1 -> 5x49x4 + 4x4x9 + 4x2x9 + 2x2x9 + 2x1x9 + =  20*49 + (16 + 8 + 4 + 2)*9  = 980 + 270 = 1250
    # 8, 8, 4, 1 -> 5x49x8 + 8x8x9 + 8x4x9 + 4x4x9 + 4x1x9 = 40*49 + (64 + 32 + 16 + 4)*9 = 1960 + 1044 = 4004
    # 16, 16, 8, 1 -> 5x49x16 + 16x16x9 + 16x8x9 + 8x8x9 + 8x1x9 = 80*49 + (256 + 128 + 64 + 4)*9 = 3820 + 4068 = 7888

  def forward(self, x):
    x = self.relu(self.fc(x))
    x = x.reshape(x.shape[0], self.nc[0], 7, 7)
    x = self.upsamp(x)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.conv2(x)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

# class ConvEnc(nn.Module):
#
#   def __init__(self, d_enc, layer_spec):
#     super(ConvEnc, self).__init__()
#     assert layer_spec[-1][0] == 'f'  # last layer must be linear
#     layer_spec[-1][1][1] = d_enc  # set linear output to d_enc
#     self.layer_spec = layer_spec
#     self.layers = conv_layers_by_spec(self.layer_spec)
#
#   def forward(self, x):
#     x = conv_forward_pass_by_spec(x, self.layer_spec, self.layers)
#     return x
#
#
# class ConvDec(nn.Module):
#
#   def __init__(self, d_enc, layer_spec):
#     super(ConvDec, self).__init__()
#     assert layer_spec[0][0] == 'f'  # first layer must be linear
#     layer_spec[0][1][0] = d_enc  # set linear input to d_enc
#     self.layer_spec = layer_spec
#     self.layers = conv_layers_by_spec(self.layer_spec)
#
#   def forward(self, x):
#     x = conv_forward_pass_by_spec(x, self.layer_spec, self.layers)
#     if self.use_sigmoid:
#       x = self.sigmoid(x)
#     return x


# def conv_layers_by_spec(spec):
#   layers = []
#   for key, v in spec:
#     if key == 'c':  # format: cin-cout-kernel-stride-padding
#       layer = nn.Conv2d(v[0], v[1], v[2], stride=v[3], padding=v[4])
#     elif key == 'f':  # format: din-dout
#       layer = nn.Linear(v[0], v[1])
#     elif key == 'r':  # format: d0-d1-d2-....
#       def reshape_fun(x):
#         return pt.reshape(x, shape=[-1] + list(v))
#       layer = reshape_fun
#     elif key == 'u':  # format: scale
#       layer = nn.UpsamplingBilinear2d(scale_factor=v[0])
#     else:
#       raise KeyError
#     layers.append(layer)
#   return layers
#
#
# def conv_forward_pass_by_spec(x, spec, layers):
#   for idx, (s, l) in enumerate(zip(spec, layers)):
#     l(x)
#     if (s[0] == 'f' or s[0] == 'c') and idx+1 < len(layers):
#       x = nn.functional.relu(x)
#   return x
#
#
# def default_cnn_specs():
#   #  1-2-4-4 filters
#   enc = 'c1-2-3-1-1,c2-4-4-2-1,c4-4-3-1-1,c4-4-4-2-1,r196,f196-0'
#   dec = 'f0-196,r4-7-7,c4-4-3-1-1,u2,c4-2-3-1-1,c2-2-3-1-1,u2,c2-1-3-1-1'
#
#   #  1-4-8-8 filters
#   enc = 'c1-4-3-1-1,c4-8-4-2-1,c8-8-3-1-1,c8-8-4-2-1,r392,f392-0'
#   dec = 'f0-392,r8-7-7,c8-8-3-1-1,u2,c8-4-3-1-1,c4-4-3-1-1,u2,c4-1-3-1-1'
#
#   #  1-4-8-16 filters
#   enc = 'c1-4-3-1-1,c4-8-4-2-1,c8-8-3-1-1,c8-16-4-2-1,r786,f786-0'
#   dec = 'f0-786,r16-7-7,c16-8-3-1-1,u2,c8-4-3-1-1,c4-4-3-1-1,u2,c4-1-3-1-1'
