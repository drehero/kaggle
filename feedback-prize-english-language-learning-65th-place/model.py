import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class ConcatenatePooling(nn.Module):
    """
    Pooling layer which concatenates the last_n layers of a model.
    """
    def __init__(self, last_n):
        super().__init__()
        self.last_n = last_n

    def forward(self, all_hidden_states):
        """
        Takes in all the hidden states of a transformer output
        and concatenates the last_n hidden states of the sequence to a
        (hidden_size * last_n) x seq_len tensor.
        """
        all_hidden_states = torch.stack(all_hidden_states)
        concatenate_pooling = torch.cat([all_hidden_states[-(i+1)] for i in range(self.last_n)], -1)
        return concatenate_pooling


class MeanConcatenatePooling(nn.Module):
    """
    Pooling layer which concatenates the last_n layers and then performs mean pooling.
    """
    def __init__(self, last_n):
        super().__init__()
        self.concatenate_pooling = ConcatenatePooling(last_n)
        self.mean_pooling = MeanPooling()

    def forward(self, all_hidden_states, attention_mask):
        """
        Args: all_hidden_states - hidden states of all layers of a transformer model
        Returns: Mean pooling over the hidden states of the last n layers
        """
        concat = self.concatenate_pooling(all_hidden_states)
        concat_mean = self.mean_pooling(concat, attention_mask)
        return concat_mean


class Conv1DPooling(nn.Module):
    def __init__(self, last_n, hidden_size):
        super().__init__()
        self.concatenate_pooling = ConcatenatePooling(last_n)
        conv_in_dim = hidden_size * last_n
        self.cnn1 = nn.Conv1d(conv_in_dim, conv_in_dim // 2, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(conv_in_dim // 2, conv_in_dim // 4, kernel_size=2, padding=0)
        self.mean_pooling = MeanPooling()

    def forward(self, all_hidden_states, attention_mask):
        concat = self.concatenate_pooling(all_hidden_states).permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(concat))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        mean_embedding = self.mean_pooling(cnn_embeddings.permute(0, 2, 1), attention_mask)
        return mean_embedding


class WeightedLayerPooling(nn.Module):
    def __init__(self, last_n, layer_weights=None):
        super().__init__()
        self.last_n = last_n
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * last_n, dtype=torch.float)
            )
        self.mean_pooling = MeanPooling()

    def forward(self, all_hidden_states, attention_mask):
        # TODO use attention_mask to compute average (s above)
        all_layer_embeddings = torch.stack(all_hidden_states)[-self.last_n:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embeddings.size())
        weighted_average = (weight_factor*all_layer_embeddings).sum(dim=0) / self.layer_weights.sum()
        mean_weighted_average = self.mean_pooling(weighted_average, attention_mask)
        return mean_weighted_average


class MLP(nn.Module):
    # https://github.com/georgian-io/Multimodal-Toolkit
    def __init__(
        self,
        in_dim,
        out_dim,
        num_hidden_layers=2,
        dropout_prob=0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()
        self.layer_channels = [in_dim]
        for _ in range(num_hidden_layers):
            self.layer_channels += [max(self.layer_channels[-1] // 2, 1)]
        self.layer_channels += [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_channels[i], self.layer_channels[i+1]) for i in range(len(self.layer_channels)-2)]
        )
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.layers += [final_layer]
        self.ln = nn.ModuleList([torch.nn.LayerNorm(dim) for dim in self.layer_channels[1:-1]])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if layer == self.layers[-1]:
                x = layer(x)
            else:
                x = self.activation(self.ln[i](layer(x)))
        return x


class Net(nn.Module):
    def __init__(self, args, config_path=None, pretrained=True):
        super().__init__()
        self.args = args

        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                args.model, output_hidden_states=True
            )
            if args.loss != "rdrop":
                self.config.hidden_dropout = args.hidden_dropout
                self.config.hidden_dropout_prob = args.hidden_dropout_prob
                self.config.attention_dropout = args.attention_dropout
                self.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(args.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # layers
        if args.token_level_feats:
            # https://github.com/fhamborg/NewsMTSC/blob/f8869bb570f4d6e809beafd97389d91e525aca77/NewsSentiment/models/singletarget/grutscsingle.py
            self.knowledge_embeddings = nn.Linear(
                self.args.n_knowledge_dims,
                self.config.hidden_size
            )
            self._init_weights(self.knowledge_embeddings)
            self.gru = nn.GRU(
                self.config.hidden_size * 2,
                self.config.hidden_size * 2,
                bidirectional=True,
                batch_first=True
            )
            self.fc = nn.Linear(
				# 3 inputs (original last gru out, mean, max), 2 inputs to gru (bert and
				# knowledge embedding), 2 (because bidirectional gru)
				self.config.hidden_size * 3 * 2 * 2,
				len(args.target_names),
            ) 
            
        elif args.pooling == "mean":
            self.pool = MeanPooling()
            self.fc = nn.Linear(self.config.hidden_size, len(args.target_names))

        elif args.pooling.startswith("mean-concat"):
            last_n = int(args.pooling.split("-")[-1])
            self.pool = MeanConcatenatePooling(last_n) 
            self.fc = nn.Linear(self.config.hidden_size * last_n, len(args.target_names))
        elif args.pooling.startswith("conv1d"):
            last_n = int(args.pooling.split("-")[-1])
            self.pool = Conv1DPooling(last_n, self.config.hidden_size)
            self.fc = nn.Linear((self.config.hidden_size*last_n)//4, len(args.target_names))
        elif args.pooling.startswith("weighted-layers"):
            last_n = int(args.pooling.split("-")[-1])
            self.pool = WeightedLayerPooling(last_n)
            self.fc = nn.Linear(self.config.hidden_size, len(args.target_names))

        if args.reinit_layers > 0:
            for layer in self.model.encoder.layer[-args.reinit_layers: ]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0, std=self.config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0, std=self.config.initializer_range)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.)

        if args.essay_level_feats:
            mlp_out_dims = self.args.essay_level_feats_dim // 2
            self.mlp = MLP(
                self.args.essay_level_feats_dim,
                mlp_out_dims
            )
            self.fc = nn.Linear(self.config.hidden_size + mlp_out_dims, len(args.target_names))
            self._init_weights(self.mlp)

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.)


    def forward(self, inputs):
        lm_outputs = self.model(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"]
        )
        if self.args.pooling not in ["mean", "none"]:
            hidden_states = lm_outputs[1]
        else: 
            hidden_states = lm_outputs[0]
        if not self.args.token_level_feats:
            feature = self.pool(hidden_states, inputs["attention_mask"])
        else:
            # token level features are combined with lm output using a bigru
            # https://github.com/fhamborg/NewsMTSC/blob/f8869bb570f4d6e809beafd97389d91e525aca77/NewsSentiment/models/singletarget/grutscsingle.py#L74
            token_level_feats = self.knowledge_embeddings(
                inputs["token_level_feats"]
            )
            lm_and_knowledge = torch.cat(
                (hidden_states, token_level_feats), dim=2
            )
            gru_all_hidden, gru_last_hidden = self.gru(
                lm_and_knowledge,
            )
            gru_last_hidden_dir0 = gru_last_hidden[0, :, :]
            gru_last_hidden_dir1 = gru_last_hidden[1, :, :]
            gru_last_hidden_stacked = torch.cat(
                (gru_last_hidden_dir0, gru_last_hidden_dir1), dim=1
            )
            # pooling
            gru_avg = torch.mean(gru_all_hidden, dim=1)
            gru_max, _ = torch.max(gru_all_hidden, dim=1)
            feature = torch.cat(
                (gru_last_hidden_stacked, gru_avg, gru_max), dim=1
            )

        if self.args.essay_level_feats:
            essay_level_feats = self.mlp(inputs["essay_level_feats"])
            feature = torch.cat((feature, essay_level_feats), dim=1)
        output = self.fc(feature)
        return output


class SegScaleModel(nn.Module):
    # https://github.com/AndriyMulyar/bert_document_classification/blob/master/bert_document_classification/document_bert_architectures.py
    # https://github.com/lingochamp/Multi-Scale-BERT-AES/blob/main/document_bert_architectures.py
    def __init__(self, args, config_path=None, pretrained=True):
        super().__init__()
        self.args = args
        if config_path == None:
            self.config = AutoConfig.from_pretrained(
                args.model, output_hidden_states=True
            )
            if args.loss != "rdrop":
                self.config.hidden_dropout = args.hidden_dropout
                self.config.hidden_dropout_prob = args.hidden_dropout_prob
                self.config.attention_dropout = args.attention_dropout
                self.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(args.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.config.hidden_size, len(args.target_names))
        )
        self.w_omega = nn.Parameter(torch.Tensor(self.config.hidden_size, self.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)

        self._init_weights(self.mlp)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1)


    def forward(self, document_batch):#, lm_batch_size):
        lm_batch_size = document_batch.shape[1]
        lm_output = torch.zeros(
            (
                document_batch.shape[0],
                #min(document_batch.shape[1], lm_batch_size),
                lm_batch_size,
                self.config.hidden_size
            ), dtype=torch.float, device=self.args.device
        )
        for doc_id in range(document_batch.shape[0]):
            lm_output[doc_id][:lm_batch_size] = self.model(
                input_ids=document_batch[doc_id][:lm_batch_size, 0],
                token_type_ids=document_batch[doc_id][:lm_batch_size, 1],
                attention_mask=document_batch[doc_id][:lm_batch_size, 2]
            )[0][:, 0, :]
        output, (_, _) = self.lstm(lm_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        # (batch_size, seq_len, num_hidden)
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # (batch_size, num_hiddens)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class WordDocScaleModel(nn.Module):
    def __init__(self, args, config_path=None, pretrained=True):
        super().__init__()
        self.args = args
        if config_path == None:
            self.config = AutoConfig.from_pretrained(
                args.model, output_hidden_states=True
            )
            if args.loss != "rdrop":
                self.config.hidden_dropout = args.hidden_dropout
                self.config.hidden_dropout_prob = args.hidden_dropout_prob
                self.config.attention_dropout = args.attention_dropout
                self.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(args.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(p=0.1)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.config.hidden_size*2, len(self.args.target_names))
        )
        self._init_weights(self.mlp)
   

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1)


    def forward(self, document_batch):
        all_lm_output_info = self.model(
            input_ids=document_batch[:, :1, 0].squeeze(),
            token_type_ids=document_batch[:, :1, 1].squeeze(),
            attention_mask=document_batch[:, :1, 2].squeeze(),
        )
        lm_token_max = torch.max(all_lm_output_info[0], 1)
        cls_token = all_lm_output_info[0][:, 0, :]
        #lm_output[doc_id][:1] = torch.cat((lm_token_max.values, all_lm_output_info[1]), 1)
        lm_output = torch.cat((lm_token_max.values, cls_token), 1)

        #lm_output = torch.zeros(
        #    (document_batch.shape[0], 1, self.config.hidden_size*2),
        #    dtype=torch.float,
        #    device=self.args.device
        #)
        #for doc_id in range(document_batch.shape[0]):
        #    all_lm_output_info = self.model(
        #        input_ids=document_batch[doc_id][:1, 0],
        #        token_type_ids=document_batch[doc_id][:1, 1],
        #        attention_mask=document_batch[doc_id][:1, 2],
        #    )
        #    lm_token_max = torch.max(all_lm_output_info[0], 1)
        #    cls_token = all_lm_output_info[0][:, 0, :]
        #    #lm_output[doc_id][:1] = torch.cat((lm_token_max.values, all_lm_output_info[1]), 1)
        #    lm_output[doc_id][:1] = torch.cat((lm_token_max.values, cls_token), 1)
        #prediction = self.mlp(lm_output.view(lm_output.shape[0], -1))
        prediction = self.mlp(lm_output)
        return prediction


class MultiScaleModel(nn.Module):
    def __init__(self, args, config_path=None, pretrained=True):
        super().__init__()
        self.args = args
        self.word_doc_model = WordDocScaleModel(args, config_path, pretrained)
        self.seg_model = SegScaleModel(args, config_path, pretrained)
        self.word_doc_model.to(args.device)
        self.seg_model.to(args.device)


    def forward(self, inputs):
        preds = self.word_doc_model(inputs["word_doc"]).squeeze()
        for chunk_idx in range(len(self.args.chunk_sizes)):
            seg_batch = inputs["seg_chunks"][chunk_idx]
            chunk_preds = self.seg_model(seg_batch).squeeze()#, self.args.lm_batch_sizes[chunk_idx]).squeeze()
            preds += chunk_preds
        return preds
