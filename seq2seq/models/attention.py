import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output
        method(str): The method to compute the alignment, mlp or dot

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
        method (torch.nn.Module): layer that implements the method of computing the attention vector

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = torch.randn(5, 3, 256)
         >>> output = torch.randn(5, 5, 256)
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim, method, level=0):
        super(Attention, self).__init__()
        self.mask = None
        self.method = self.get_method(method, dim, level)
        self.last_attention = []

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, decoder_states, encoder_states, **attention_method_kwargs):

        batch_size = decoder_states.size(0)
        decoder_states_size = decoder_states.size(2)
        input_size = encoder_states.size(1)

        # compute mask
        mask = encoder_states.eq(0.)[:, :, :1].transpose(1, 2)

        # Compute attention vals
        attn = self.method(decoder_states, encoder_states, **attention_method_kwargs)

        # Preparation only needed if the attention doesn't come from the baseline
        if not isinstance(self.method, BaselineGuidance):
            if self.mask is not None:
                attn.masked_fill_(self.mask, -float('inf'))

            # apply local mask
            attn.masked_fill_(mask, -float('inf'))

            attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # Diffuse attention after softmax is applied
        if isinstance(self.method, DiffusedGuidance):
            attn = self.method.diffuse(attn)

        self.last_attention.append(attn)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, encoder_states)

        return context, attn

    def get_method(self, method, dim, level=0):
        """
        Set method to compute attention
        """
        if method == 'mlp':
            method = MLP(dim)
        elif method == 'concat':
            method = Concat(dim)
        elif method == 'dot':
            method = Dot()
        elif method == 'hard':
            method = HardGuidance()
        elif method == 'diffused':
            method = DiffusedGuidance(level)
        elif method == "baseline":
            method = BaselineGuidance()
        else:
            raise ValueError("Unknown attention method")

        return method

    def clean_memory(self):
        self.last_attention = []


class Concat(nn.Module):
    """
    Implements the computation of attention by applying an
    MLP to the concatenation of the decoder and encoder
    hidden states.
    """

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.mlp = nn.Linear(dim * 2, 1)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _,          dec_seqlen, _       = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and respape to get in correct form
        mlp_output = self.mlp(mlp_input)
        attn = mlp_output.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class Dot(nn.Module):

    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, decoder_states, encoder_states):
        attn = torch.bmm(decoder_states, encoder_states.transpose(1, 2))
        return attn


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(dim * 2, dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(dim, 1)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _,          dec_seqlen, _       = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and reshape to get in correct form
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.activation(mlp_output)
        out = self.out(mlp_output)
        attn = out.view(batch_size, dec_seqlen, enc_seqlen)

        return attn

class HardGuidance(nn.Module):
    """
    Attention method / attentive guidance method for data sets that are annotated with attentive guidance.
    """

    def forward(self, decoder_states, encoder_states, step, provided_attention):
        """
        Forward method that receives provided attentive guidance indices and returns proper
        attention scores vectors.

        Args:
            decoder_states (torch.FloatTensor): Hidden layer of all decoder states (batch, dec_seqlen, hl_size)
            encoder_states (torch.FloatTensor): Output layer of all encoder states (batch, dec_seqlen, hl_size)
            step (int): The current decoder step for unrolled RNN. Set to -1 for rolled RNN
            provided_attention (torch.LongTensor): Variable containing the provided attentive guidance indices (batch, max_provided_attention_length)

        Returns:
            torch.tensor: Attention score vectors (batch, dec_seqlen, hl_size)
        """

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, _ = encoder_states.size()
        _,          dec_seqlen, _ = decoder_states.size()

        attention_indices = provided_attention.detach()
        # If we have shorter examples in a batch, attend the PAD outputs to the first encoder state
        attention_indices.masked_fill_(attention_indices.eq(-1), 0)

        # In the case of unrolled RNN, select only one column
        if step != -1:
            attention_indices = attention_indices[:, step]

        # Add a (second and) third dimension
        # In the case of rolled RNN: (batch_size x dec_seqlen) -> (batch_size x dec_seqlen x 1)
        # In the case of unrolled:   (batch_size)              -> (batch_size x 1          x 1)
        attention_indices = attention_indices.contiguous().view(batch_size, -1, 1)
        # Initialize attention vectors. These are the pre-softmax scores, so any
        # -inf will become 0 (if there is at least one value not -inf)
        attention_scores = torch.full([batch_size, dec_seqlen, enc_seqlen], fill_value=-float('inf'), device=device)
        attention_scores = attention_scores.scatter_(dim=2, index=attention_indices, value=1)
        attention_scores = attention_scores
 
        return attention_scores


class DiffusedGuidance(nn.Module):
    """
    Attention method that takes hard guidance from a data set and
    diffuses it according to a level indicated.
    0 means no diffusion
    1 means the original attention position gets no attention at all
    -1 means uniform
    """
    def __init__(self, level):
        super(DiffusedGuidance, self).__init__()
        self.level = level
        self.method = HardGuidance()

    def forward(self, decoder_states, encoder_states, step, provided_attention):
        # Get the attention as calculated by the hard guidance model
        return self.method(decoder_states, encoder_states, step, provided_attention)

    def diffuse(self, attention):
        for i in range(attention.shape[2]-1):
            # Uniform attention
            if self.level == -1:
                attention[0][0][i] = 1 / (attention.shape[2] - 1)
            # Diffuse the hard attention
            else:
                if attention[0][0][i].item() == 1:
                    attention[0][0][i] = 1 - self.level
                else:
                    attention[0][0][i] = self.level / (attention.shape[2] - 2)
        return attention


class BaselineGuidance(nn.Module):
    """
    Attention method that transfers attention of the baseline model to 
    the model currently used.
    """

    def forward(self, decoder_states, encoder_states, provided_attention):
        return provided_attention
