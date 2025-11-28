import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import agent.dreamer_utils as common
import agent.video_utils as video_utils
from tools.genrl_utils import viclip_global_instance

Module = nn.Module

class ThinkerAgent(Module):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.cfg.update(**kwargs)
        self.device = cfg.device
        self.act_spec = act_spec # 无效
        
        self.n_frames = 8 
        
        if 'clip_video' in obs_space:
            self.viclip_emb_dim = obs_space['clip_video'].shape[0]
        else:
            self.viclip_emb_dim = 512 

        self.wm = VideoWorldModel(cfg, obs_space, self.viclip_emb_dim, self.n_frames)
        
        self.to(self.device)
        self.requires_grad_(requires_grad=False)

    def act(self, obs, meta, step, eval_mode, state):
        B = len(obs['is_first'])
        action = torch.zeros((B,) + self.act_spec.shape, device=self.device)
        return action.cpu().numpy()[0], state

    def update_wm(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        return state, outputs, metrics

    def update(self, data, step):
        return self.update_wm(data, step)

    def report(self, data, nvid=8):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads['decoder'].cnn_keys:
            name = key.replace('/', '_')
            report[f'openl_{name}'] = self.wm.video_pred(data, key, nvid=nvid)
        return report

    def init_meta(self): return OrderedDict()
    def update_meta(self, meta, *args): return meta


class VideoWorldModel(Module):
    def __init__(self, config, obs_space, viclip_emb_dim, n_frames):
        super().__init__()
        self.cfg = config
        self.device = config.device
        self._use_amp = (config.precision == 16)
        self.n_frames = n_frames
        self.viclip_emb_dim = viclip_emb_dim
        
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        
        # 1. Encoder
        self.encoder = common.Encoder(shapes, **config.encoder)
        with torch.no_grad():
            zeros = {k: torch.zeros((1,) + v) for k, v in shapes.items()}
            embed_dim = self.encoder(zeros).shape[1]
        self.embed_dim = embed_dim

        # 2. VideoSSM
        rssm_action_dim = viclip_emb_dim + n_frames
        
        self.rssm = video_utils.VideoSSM(
            **config.connector,      # 传入 connector 参数
            **config.connector_rssm, # 传入 RSSM 结构参数
            connector_kl=config.connector_kl,
            n_frames=n_frames,
            action_dim=rssm_action_dim,
            clip_add_noise=self.cfg.clip_add_noise,
            clip_lafite_noise=self.cfg.clip_lafite_noise,
            embed_dim=embed_dim,     
            device=self.device,
            cell_input='stoch',
            use_obs_model=True      
        )
        
        # 3. Decoder
        c_conf = config.connector_rssm
        deter_size = c_conf.deter
        if c_conf.get('discrete', 0) > 0:
            stoch_size = c_conf.stoch * c_conf.discrete
        else:
            stoch_size = c_conf.stoch
            
        self.decoder_input_type = config.decoder_inputs 
        
        if self.decoder_input_type == 'stoch':
            self.decoder_input_size = stoch_size
            self.get_decoder_input = self.rssm.get_stoch
        elif self.decoder_input_type == 'deter':
            self.decoder_input_size = deter_size
            self.get_decoder_input = self.rssm.get_deter
        else: # 'feat'
            self.decoder_input_size = stoch_size + deter_size
            self.get_decoder_input = self.rssm.get_feat

        self.heads = nn.ModuleDict()
        self.heads['decoder'] = common.Decoder(
            shapes, **config.decoder, 
            embed_dim=self.decoder_input_size, 
            image_dist=config.image_dist
        )

        # 4. ViCLIP
        if not getattr(config, "viclip_encode", False):
            if not viclip_global_instance._instantiated:
                viclip_global_instance.instantiate(device=self.device)
            self.viclip_model = viclip_global_instance.viclip
            self.viclip_model.requires_grad_(False)

        # 5. Optimizer
        self.model_opt = common.Optimizer(
            'model', self.parameters(), 
            **config.model_opt, 
            use_amp=self._use_amp
        )
        self.eval()

    def _get_video_embed(self, data):
        B, T = data['observation'].shape[:2]
        
        if getattr(self.cfg, "viclip_encode", False):
            return data['clip_video']
        else:
            with torch.no_grad():
                obs = data['observation'] / 255.0
                processed_obs = self.viclip_model.preprocess_transf(obs.reshape(B*T, *obs.shape[2:]))
                chunks = T // self.n_frames
                reshaped_obs = processed_obs.reshape(B * chunks, self.n_frames, 3, 224, 224)

                video_feat = self.viclip_model.get_vid_features(reshaped_obs.to(self.viclip_model.device))

                video_feat = video_feat.reshape(B, chunks, -1)
                video_feat = video_feat.unsqueeze(2).repeat(1, 1, self.n_frames, 1)
                video_feat = video_feat.reshape(B, T, -1)
                return video_feat

    def update(self, data, state=None):
        self.train()
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss, self.parameters()))
        self.eval()
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        B, T = data['observation'].shape[:2]
        
        # --- 1. 准备 Video Action ---
        video_embed = self._get_video_embed(data)
        
        # 切片：只取每个 Chunk 的最后一帧 (GenRL 逻辑)
        indices = torch.arange(self.n_frames - 1, T, self.n_frames, device=video_embed.device)
        video_embed_chunk = torch.index_select(video_embed, 1, indices) 
        
        # 扩展回全长
        video_embed_full = video_embed_chunk.unsqueeze(2).repeat(1, 1, self.n_frames, 1).reshape(B, T, -1)
        
        orig_video_embed = video_embed_full.clone()

        # 加噪 (Slice -> Repeat -> Add Noise 顺序，保证 Variational Noise)
        if self.training:
            if self.rssm.clip_add_noise > 0:
                noise = torch.randn_like(video_embed_full) * self.rssm.clip_add_noise
                video_embed_full = video_embed_full + noise
                video_embed_full = F.normalize(video_embed_full, dim=-1)
            
            if self.rssm.clip_lafite_noise > 0:
                noise = torch.randn_like(video_embed_full)
                normed_noise = F.normalize(noise, dim=-1)
                video_embed_full = (1 - self.rssm.clip_lafite_noise) * video_embed_full + \
                                   self.rssm.clip_lafite_noise * normed_noise
                video_embed_full = F.normalize(video_embed_full, dim=-1)

        # DAE 逻辑
        denoising_loss = 0.0
        if self.rssm.denoising_ae:
            if (self.rssm.clip_lafite_noise + self.rssm.clip_add_noise) > 0:
                denoised_embed = self.rssm.aligner(video_embed_full)
                denoised_embed = F.normalize(denoised_embed, dim=-1)
                denoising_loss = 1 - F.cosine_similarity(denoised_embed, orig_video_embed, dim=-1).mean()
            
            # 如果开启 DAE，RSSM 接收 Clean 的 Action
            video_embed_full = orig_video_embed

        # 获取最终 Action (含时间编码)
        embed_action = self.rssm.get_action(video_embed_full)

        # --- 2. Encoder & RSSM Observe ---
        embed = self.encoder(data)
        
        if state is None:
            # 使用第一帧的 Action 初始化
            state = self.rssm.initial(B, init_embed=embed_action[:, 0])
            
        post, prior = self.rssm.observe(embed, embed_action, data['is_first'], state)

        # --- 3. Losses ---
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.connector_kl)
        
        decoder_input = self.get_decoder_input(post)
        out = self.heads['decoder'](decoder_input)
        
        likes = {}
        dists = out if isinstance(out, dict) else {'observation': out}
        for key, dist in dists.items():
            if key in data:
                likes[key] = dist.log_prob(data[key])
        
        recon_loss = sum(-l.mean() for k, l in likes.items())
        kl_scale = self.cfg.loss_scales.get('kl', 1.0)
        
        model_loss = recon_loss + kl_scale * kl_loss + denoising_loss
        
        # Metrics
        outs = dict(embed=embed, post=post, prior=prior, likes=likes, kl=kl_value)
        metrics = {f'{k}_loss': -v.mean().item() for k, v in likes.items()}
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        if self.rssm.denoising_ae:
            metrics['aligner_loss'] = denoising_loss.item() if isinstance(denoising_loss, torch.Tensor) else 0
        
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def video_pred(self, data, key, nvid=8):
        decoder = self.heads['decoder']
        truth = data[key][:nvid] + 0.5
        data = self.preprocess(data)
        embed = self.encoder(data)
        B, T = data['observation'].shape[:2]
        
        # Inference: Get Clean Video Embed -> Slice -> Repeat -> Get Action
        video_embed = self._get_video_embed(data)
        indices = torch.arange(self.n_frames - 1, T, self.n_frames, device=video_embed.device)
        video_embed_chunk = torch.index_select(video_embed, 1, indices)
        video_embed_full = video_embed_chunk.unsqueeze(2).repeat(1, 1, self.n_frames, 1).reshape(B, T, -1)
        embed_action = self.rssm.get_action(video_embed_full)
        
        # Observe (Reconstruction)
        init_state = self.rssm.initial(nvid, init_embed=embed_action[:nvid, 0])
        states, _ = self.rssm.observe(
            embed[:nvid, :5], 
            embed_action[:nvid, :5], 
            data['is_first'][:nvid, :5],
            state=init_state
        )
        
        recon_input = self.get_decoder_input(states)
        recon = decoder(recon_input)[key].mean
        
        # Imagine (Prediction)
        init = {k: v[:, -1] for k, v in states.items()}
        # 使用后续帧的 Action 进行预测
        action_pred = embed_action[:nvid, 5:]
        prior = self.rssm.imagine(action_pred, init)
        
        prior_input = self.get_decoder_input(prior)
        prior_recon = decoder(prior_input)[key].mean
        
        model = torch.clip(torch.cat([recon + 0.5, prior_recon + 0.5], 1), 0, 1)
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)
        return video

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_'): continue
            if value.dtype in [np.uint8, torch.uint8]: value = value / 255.0 - 0.5 
            obs[key] = value
        return obs