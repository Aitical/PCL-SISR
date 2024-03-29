from model import nlsn_common as common
from model import nlsn_util as attention
import torch.nn as nn

def make_model(args, parent=False):
    if args.nlsn_dilation:
        from model import dilated
        return NLSN(args, dilated.dilated_conv)
    else:
        return NLSN(args)


class NLSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLSN, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=args.nlsn_chunk_size, n_hashes=args.nlsn_n_hashes, reduction=4, res_scale=args.res_scale)]         

        for i in range(n_resblock):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
            if (i+1)%8==0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=args.nlsn_chunk_size, n_hashes=args.nlsn_n_hashes, reduction=4, res_scale=args.res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 
