# %%
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math
# %%


def norm_layer(channel_in, groups=None):
    if groups is None:
        return nn.BatchNorm2d(channel_in)
    else:
        return nn.GroupNorm(groups, channel_in)


class AttentionBlock(nn.Module):
    """Applies self-attention.
    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=None):
        super(AttentionBlock, self).__init__()
        self.units = units  # number of channels
        self.norm = norm_layer(units, groups)
        self.query = nn.Linear(units, units)
        self.key = nn.Linear(units, units)
        self.value = nn.Linear(units, units)
        self.proj = nn.Linear(units, units)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        scale = self.units ** (-0.5)
        # [b, c, h, w] -> [b, h, w, c]
        inputs = self.norm(inputs)  # [b, c, h, w]
        # [b, c, h, w] -> [b, h, w, c]
        inputs = inputs.permute(0, 2, 3, 1)
        q = self.query(inputs)  # [b, h, w, c]
        k = self.key(inputs)  # [b, h, w, c]
        v = self.value(inputs)  # [b, h, w, c]

        # equivalent: [hw X C] * [hw X C]^T, eliminate the index of c
        attn_score = torch.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = attn_score.view(batch_size, height, width, height * width)

        attn_score = nnF.softmax(attn_score, dim=-1)  # [b, h, w, hw]
        attn_score = attn_score.view(
            batch_size, height, width, height, width
        )  # [b, h, w, h, w]

        # equivalent: [hw X hw] * [hw X c]
        proj = torch.einsum("bhwHW,bHWc->bhwc", attn_score, v)  # [b, h, w, c]
        proj = self.proj(proj)  # [b, h, w, c]
        # [b, h, w, c] -> [b, c, h, w]
        output = (inputs + proj).permute(0, 3, 1, 2)
        return output


# %%


class ResidualBlockConv(nn.Module):
    """Residual block.

    Args:
        channel: Number of channels in the convolutional layers
        groups: Number of groups to be used for GroupNormalization layer
        activation_fn: Activation function to be used
    """

    def __init__(
        self, channel_in, channel_out, groups=8, activation_fn=nn.SiLU(), dropout=None
    ):
        super(ResidualBlockConv, self).__init__()

        self.activation_fn = activation_fn
        self.net = nn.ModuleList()
        self.norm = norm_layer(channel_in, groups)
        self.net.append(activation_fn)
        if dropout is not None:
            self.net.append(nn.Dropout2d(p=dropout))
        self.net.append(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding="same")
        )
        if groups is None:
            self.net.append(nn.BatchNorm2d(channel_out))
        else:
            self.net.append(nn.GroupNorm(groups, channel_out))
        self.net.append(activation_fn)
        if dropout is not None:
            self.net.append(nn.Dropout2d(p=dropout))
        self.net.append(
            nn.Conv2d(channel_out, channel_out, kernel_size=3, padding="same")
        )
        if channel_in != channel_out:
            self.shortcut = nn.Conv2d(channel_in, channel_out, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        x = inputs
        for layer in self.net:
            x = layer(x)  # [b,ci,h,w]->[b,co,h,w]
        x = x + self.shortcut(inputs)
        return x


# %%


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.channel = channel
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        return self.conv(inputs)


class UpSample(nn.Module):
    def __init__(self, channel, interpolation="nearest"):
        super(UpSample, self).__init__()
        self.channel = channel
        self.interpolation = interpolation
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        return x


# %%


class UNet(nn.Module):
    def __init__(
        self,
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention=None,
        num_res_blocks=1,
        norm_groups=None,
        interpolation="nearest",
        activation_fn=nn.SiLU(),
        dropout=None,
    ):
        super().__init__()
        channel_in, img_size = img_shape[0], img_shape[1]
        channel_list = [
            first_conv_channels * channel_mutiplier
            for channel_mutiplier in channel_mutipliers
        ]
        if has_attention is None:
            has_attention = [False] * len(channel_list)

        if len(has_attention) != len(channel_list):
            raise ValueError(
                "has_attention should have the same length as channel_mutipliers"
            )

        self.activation_fn = activation_fn
        self.skip_channels = []
        encoder_skip_idx = []
        self.first_conv = nn.Conv2d(
            channel_in, first_conv_channels, kernel_size=3, padding=1
        )

        self.encoder = nn.ModuleList()
        for i in range(len(channel_list)):
            if i == 0:
                in_channel = first_conv_channels
            else:
                in_channel = channel_list[i - 1]
            self.encoder.append(
                ResidualBlockConv(
                    in_channel,
                    channel_list[i],
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            if has_attention[i]:
                self.encoder.append(AttentionBlock(
                    channel_list[i], groups=norm_groups))
            self.skip_channels.append(channel_list[i])
            encoder_skip_idx.append(len(self.encoder) - 1)
            for _ in range(1, num_res_blocks):
                self.encoder.append(
                    ResidualBlockConv(
                        channel_list[i],
                        channel_list[i],
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                if has_attention[i]:
                    self.encoder.append(
                        AttentionBlock(channel_list[i], groups=norm_groups)
                    )
                self.skip_channels.append(channel_list[i])
                encoder_skip_idx.append(len(self.encoder) - 1)
                # skip connection
            if i != len(channel_list) - 1:
                self.encoder.append(DownSample(channel_list[i]))

        self.encoder_skip = torch.zeros(len(self.encoder), dtype=bool)
        self.encoder_skip[encoder_skip_idx] = True

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(
            ResidualBlockConv(
                channel_list[-1],
                channel_list[-1],
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )
        self.bottleneck.append(AttentionBlock(
            channel_list[-1], groups=norm_groups))
        self.bottleneck.append(
            ResidualBlockConv(
                channel_list[-1],
                channel_list[-1],
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )

        self.decoder = nn.ModuleList()
        decoder_skip_idx = []
        for i in reversed(range(len(channel_list))):
            if i == len(channel_list) - 1:
                in_channel = channel_list[-1] + self.skip_channels.pop()
            else:
                in_channel = channel_list[i + 1] + self.skip_channels.pop()
            self.decoder.append(
                ResidualBlockConv(
                    in_channel,
                    channel_list[i],
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            decoder_skip_idx.append(len(self.decoder) - 1)
            if has_attention[i]:
                self.decoder.append(AttentionBlock(
                    channel_list[i], groups=norm_groups))
            for _ in range(1, num_res_blocks):
                in_channel = channel_list[i] + self.skip_channels.pop()
                self.decoder.append(
                    ResidualBlockConv(
                        in_channel,
                        channel_list[i],
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                decoder_skip_idx.append(len(self.decoder) - 1)
                if has_attention[i]:
                    self.decoder.append(
                        AttentionBlock(channel_list[i], groups=norm_groups)
                    )

            if i != 0:
                self.decoder.append(
                    UpSample(channel_list[i], interpolation=interpolation)
                )

        self.decoder_skip = torch.zeros(len(self.decoder), dtype=bool)
        self.decoder_skip[decoder_skip_idx] = True

    def forward(self, x):
        skips = []
        x = self.first_conv(x)
        for eskip, layer in zip(self.encoder_skip, self.encoder):
            x = layer(x)
            if eskip:
                skips.append(x)
        for layer in self.bottleneck:
            x = layer(x)
        for dskip, layer in zip(self.decoder_skip, self.decoder):
            if dskip:
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x)
        return x


# %%


class ResidualBlockConvTimeStep(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        time_channels,
        label_emb_dim,
        groups=None,
        activation_fn=nn.SiLU(),
        dropout=None,
    ):
        super().__init__()
        lay_list1 = [norm_layer(channel_in, groups), activation_fn]
        if dropout is not None:
            lay_list1.append(nn.Dropout(dropout))
        lay_list1.append(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding="same")
        )
        self.conv1 = nn.Sequential(*lay_list1)

        self.time_emb = nn.Sequential(
            activation_fn, nn.Linear(time_channels, channel_out)
        )

        self.label_emb = nn.Sequential(
            activation_fn, nn.Linear(label_emb_dim, channel_out)
        )

        lay_list2 = [norm_layer(channel_out, groups), activation_fn]
        if dropout is not None:
            lay_list2.append(nn.Dropout(dropout))
        lay_list2.append(
            nn.Conv2d(channel_out, channel_out, kernel_size=3, padding="same")
        )
        self.conv2 = nn.Sequential(*lay_list2)

        if channel_in != channel_out:
            self.shortcut = nn.Conv2d(channel_in, channel_out, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t, c, mask):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        `c` has shape `[batch_size, label_dim]`
        `mask` has shape `[batch_size, ]`
        """
        h = self.conv1(x)
        emb_t = self.time_emb(t)
        emb_c = self.label_emb(c) * mask[:, None]
        h += emb_t[:, :, None, None] + emb_c[:, :, None, None]
        h = self.conv2(h)

        return h + self.shortcut(x)


# %%
# Multi Attention block


class MultiAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, groups=None):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels, groups)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# %%
def timestep_embedding(timesteps, embed_dim, max_period=10000.0):
    """
    use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# %%


class UNetTimeStep(nn.Module):

    def __init__(
        self,
        img_shape,
        label_dim,
        one_hot,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_heads=4,
        num_res_blocks=1,
        norm_groups=None,
        interpolation="nearest",
        activation_fn=nn.SiLU(),
        dropout=None,
    ):
        super().__init__()
        channel_in, img_size = img_shape[0], img_shape[1]
        channel_list = [
            first_conv_channels * channel_mutiplier
            for channel_mutiplier in channel_mutipliers
        ]
        if has_attention is None:
            has_attention = [False] * len(channel_list)

        if len(has_attention) != len(channel_list):
            raise ValueError(
                "has_attention should have the same length as channel_mutipliers"
            )

        self.activation_fn = activation_fn
        self.skip_channels = []
        encoder_skip_idx = []
        self.first_conv_channels = first_conv_channels
        time_emb_dim = first_conv_channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(first_conv_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # labels embedding
        label_emb_dim = first_conv_channels
        if one_hot:
            self.label_emb = nn.Embedding(label_dim, label_emb_dim)
        else:
            self.label_emb = nn.Sequential(
                nn.Linear(label_dim, 2*label_emb_dim),
                activation_fn,
                nn.Linear(2*label_emb_dim, 2*label_emb_dim),
                activation_fn,
                nn.Linear(2*label_emb_dim, 2*label_emb_dim),
                activation_fn,
                nn.Linear(2*label_emb_dim, label_emb_dim),
            )

        self.first_conv = nn.Conv2d(
            channel_in, first_conv_channels, kernel_size=3, padding=1
        )

        self.encoder = nn.ModuleList()

        for i in range(len(channel_list)):
            if i == 0:
                in_channel = first_conv_channels
            else:
                in_channel = channel_list[i - 1]
            self.encoder.append(
                ResidualBlockConvTimeStep(
                    in_channel,
                    channel_list[i],
                    time_emb_dim,
                    label_emb_dim,
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            if has_attention[i]:
                self.encoder.append(
                    MultiAttentionBlock(
                        channel_list[i], num_heads=num_heads, groups=norm_groups
                    )
                )
            self.skip_channels.append(channel_list[i])
            encoder_skip_idx.append(len(self.encoder) - 1)

            for _ in range(1, num_res_blocks):
                self.encoder.append(
                    ResidualBlockConvTimeStep(
                        channel_list[i],
                        channel_list[i],
                        time_emb_dim,
                        label_emb_dim,
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                if has_attention[i]:
                    self.encoder.append(
                        MultiAttentionBlock(
                            channel_list[i], num_heads=num_heads, groups=norm_groups
                        )
                    )
                self.skip_channels.append(channel_list[i])
                encoder_skip_idx.append(len(self.encoder) - 1)
                # skip connection
            if i != len(channel_list) - 1:
                self.encoder.append(DownSample(channel_list[i]))

        self.encoder_skip = torch.zeros(len(self.encoder), dtype=bool)
        self.encoder_skip[encoder_skip_idx] = True

        self.bottleneck = nn.ModuleList()

        self.bottleneck.append(
            ResidualBlockConvTimeStep(
                channel_list[-1],
                channel_list[-1],
                time_emb_dim,
                label_emb_dim,
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )
        self.bottleneck.append(
            MultiAttentionBlock(
                channel_list[-1], num_heads=num_heads, groups=norm_groups
            )
        )
        self.bottleneck.append(
            ResidualBlockConvTimeStep(
                channel_list[-1],
                channel_list[-1],
                time_emb_dim,
                label_emb_dim,
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )

        self.decoder = nn.ModuleList()
        decoder_skip_idx = []
        for i in reversed(range(len(channel_list))):
            if i == len(channel_list) - 1:
                in_channel = channel_list[-1] + self.skip_channels.pop()
            else:
                in_channel = channel_list[i + 1] + self.skip_channels.pop()
            self.decoder.append(
                ResidualBlockConvTimeStep(
                    in_channel,
                    channel_list[i],
                    time_emb_dim,
                    label_emb_dim,
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            decoder_skip_idx.append(len(self.decoder) - 1)
            if has_attention[i]:
                self.decoder.append(
                    MultiAttentionBlock(
                        channel_list[i], num_heads=num_heads, groups=norm_groups
                    )
                )
            for _ in range(1, num_res_blocks):
                in_channel = channel_list[i] + self.skip_channels.pop()
                self.decoder.append(
                    ResidualBlockConvTimeStep(
                        in_channel,
                        channel_list[i],
                        time_emb_dim,
                        label_emb_dim,
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                decoder_skip_idx.append(len(self.decoder) - 1)
                if has_attention[i]:
                    self.decoder.append(
                        MultiAttentionBlock(
                            channel_list[i], num_heads=num_heads, groups=norm_groups
                        )
                    )

            if i != 0:
                self.decoder.append(
                    UpSample(channel_list[i], interpolation=interpolation)
                )

        self.decoder_skip = torch.zeros(len(self.decoder), dtype=bool)
        self.decoder_skip[decoder_skip_idx] = True

        self.out = nn.Sequential(
            norm_layer(channel_list[0], norm_groups),
            nn.SiLU(),
            nn.Conv2d(channel_list[0], channel_in, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, c, mask):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c: a 1-D batch of classes.
        :param mask: a 1-D batch of conditioned/unconditioned.
        :return: an [N x C x ...] Tensor of outputs.
        """
        skips = []
        x = self.first_conv(x)

        # QB: N x 1 -> N x 4*model_channels
        t_emb = self.time_emb(
            timestep_embedding(timesteps, embed_dim=self.first_conv_channels)
        )
        c_emb = self.label_emb(c)  # N x num -> N x model_channels

        for eskip, layer in zip(self.encoder_skip, self.encoder):
            if isinstance(layer, ResidualBlockConvTimeStep):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
            if eskip:
                skips.append(x)

        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlockConvTimeStep):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)

        for dskip, layer in zip(self.decoder_skip, self.decoder):
            if dskip:
                x = torch.cat([x, skips.pop()], dim=1)
            if isinstance(layer, ResidualBlockConvTimeStep):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)

        return self.out(x)
