import torch
from torch import nn

from .augmentations import GaussianSmoothing, TimeChannelMasking


class Decoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        rnn="GRU",
        use_layernorm=False,
        time_mask_p=0.2,
        n_time_masks=2,
        channel_mask_p=0.2,
        n_channel_masks=2,
    ):
        super(Decoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        self.use_layernorm = use_layernorm
        self.rnn = rnn

        self.masking = TimeChannelMasking(max_time_mask_pct=time_mask_p, num_time_masks=n_time_masks,
                                          max_channel_mask_pct=channel_mask_p, num_channel_masks=n_channel_masks)

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        if self.use_layernorm: #build layers individually + insert layernorm between each
            self.rnn_layers = nn.ModuleList()
            self.lns = nn.ModuleList()

            for i in range(layer_dim):
                input_size = (neural_dim * self.kernelLen) if i == 0 else (hidden_dim * (2 if bidirectional else 1))

                if rnn == "GRU":
                    rnn_layer = nn.GRU(
                        input_size,
                        hidden_dim,
                        num_layers=1,
                        batch_first=True,
                        dropout=0, #if using layer norm, do dropout afterwards not within the layer!
                        bidirectional=bidirectional,
                    )
                else:
                    rnn_layer = nn.LSTM(
                        input_size,
                        hidden_dim,
                        num_layers=1,
                        batch_first=True,
                        dropout=0,
                        bidirectional=bidirectional,
                    )
                
                for name, param in rnn_layer.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                
                self.rnn_layers.append(rnn_layer)

                if i < layer_dim - 1: #no layernorm after last layer
                    self.lns.append(nn.LayerNorm(hidden_dim * (2 if bidirectional else 1)))
                
            if dropout > 0: #dropout between layers
                self.dropout_layer = nn.Dropout(dropout)
            else:
                self.dropout_layer = None
        else:
            # RNN layers (either GRU or LSTM)
            self.decoder = nn.GRU(
                (neural_dim) * self.kernelLen,
                hidden_dim,
                layer_dim,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            ) if rnn == "GRU" else nn.LSTM(
                (neural_dim) * self.kernelLen,
                hidden_dim,
                layer_dim,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

            for name, param in self.decoder.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        if self.training:
            neuralInput = self.masking(neuralInput)

        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.use_layernorm:
            x = stridedInputs
            for i, rnn_layer in enumerate(self.rnn_layers):
                h0 = torch.zeros(
                    (2 if self.bidirectional else 1),
                    x.size(0),
                    self.hidden_dim,
                    device=self.device,
                ).requires_grad_()
                if self.rnn == "LSTM":
                    c0 = torch.zeros(
                        (2 if self.bidirectional else 1),
                        x.size(0),
                        self.hidden_dim,
                        device=self.device,
                    ).requires_grad_()
                    x, _ = rnn_layer(x, (h0.detach(), c0.detach()))
                else:
                    x, _ = rnn_layer(x, h0.detach())
                
                if i < len(self.lns): #no layernorm after last layer
                    x = self.lns[i](x)
                
                if self.dropout_layer is not None and i < len(self.rnn_layers) - 1:
                    x = self.dropout_layer(x)
            hid = x
        else: #original single multi-layer RNN
            if self.bidirectional:
                h0 = torch.zeros(
                    self.layer_dim * 2,
                    transformedNeural.size(0),
                    self.hidden_dim,
                    device=self.device,
                ).requires_grad_()
                if self.rnn == "LSTM":
                    c0 = torch.zeros(
                        self.layer_dim * 2,
                        transformedNeural.size(0),
                        self.hidden_dim,
                        device=self.device,
                    ).requires_grad_()
                    hid, _ = self.decoder(stridedInputs, (h0.detach(), c0.detach()))
                else:
                    hid, _ = self.decoder(stridedInputs, h0.detach())
            else:
                h0 = torch.zeros(
                    self.layer_dim,
                    transformedNeural.size(0),
                    self.hidden_dim,
                    device=self.device,
                ).requires_grad_()
                if self.rnn == "LSTM":
                    c0 = torch.zeros(
                        self.layer_dim,
                        transformedNeural.size(0),
                        self.hidden_dim,
                        device=self.device,
                    ).requires_grad_()
                    hid, _ = self.decoder(stridedInputs, (h0.detach(), c0.detach()))
                else:
                    hid, _ = self.decoder(stridedInputs, h0.detach())
        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
