import logging
import math
from re import S
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import src.utils.config as config
from src.evaluation_metrics.diversity_metrics import TopicDiversity
from src.evaluation_metrics.coherence_metrics import Coherence


class ConvNTM(nn.Module):
    def __init__(self, hidden_dim=500, l1_strength=0, w_rec_M=1.0):
        super(ConvNTM, self).__init__()
        self.input_dim = config.bow_vocab_size
        self.topic_num = config.topic_num
        topic_num = config.topic_num

        self.speaker_num = config.speaker_num
        self.use_speaker_reps = config.use_speaker_reps
        self.speaker_rep_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim, bias=False)  # matrix beta
        self.l1_strength = torch.FloatTensor([l1_strength]).to(config.device)
        self.w_rec_M = torch.FloatTensor([w_rec_M]).to(config.device)
        self.device = config.device
        # self.rep_add_bow = nn.Linear(self.speaker_rep_dim+self.input_dim, self.input_dim)
        self.global_to_local = nn.Linear(topic_num*2, topic_num)
        self.agg_rep_x1 = nn.Linear(hidden_dim+self.speaker_rep_dim, hidden_dim)
        self.agg_rep_x2 = nn.Linear(hidden_dim+self.speaker_rep_dim, hidden_dim)
        self.phi = nn.Linear(hidden_dim+self.speaker_rep_dim+hidden_dim, topic_num)
        
        self.rho = nn.Linear(hidden_dim, self.input_dim, bias=False)
        self.alphas = nn.Linear(hidden_dim, topic_num, bias=False)

    def get_beta(self):
        logit = self.fcd1.weight
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## [num_topic, vocab_size]
        return beta


    def encode(self, x, reps, speaker_uttr_num):
        x = F.normalize(x, dim=-1, p=1)
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x)) 

        e2 = torch.cat([reps, e1], dim=-1)
        e3 = torch.sigmoid(self.agg_rep_x1(e2)) * torch.tanh(self.agg_rep_x2(e2))
        bs = x.shape[0] // self.speaker_num
        mask = torch.zeros_like(e3).to(e3.device)
        for i in range(self.speaker_num):
            for b in range(bs):
                mask[bs*i+b, :speaker_uttr_num[i][b], :] = 1
        e3 = e3 * mask
        agg = torch.tanh(e3.sum(dim=1))
        phi = self.phi(torch.cat([agg.unsqueeze(1).expand(-1,e2.shape[1],-1), e2], dim=-1))
        return phi, self.fc21(agg), self.fc22(agg)


    def reparameterize(self, mu, logvar):  
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=-1)
        return d1


    def forward(self, uttr_bow, lens, uttr_reps=None, out_g=False):
        # print("uttr_bow.shape", uttr_bow.shape) # [bs*speaker_num, max_speaker_uttr_num, input_dim]
        # print("lens", lens)
        # print("uttr_reps.shape", uttr_reps.shape) # [bs*speaker_num, max_speaker_uttr_num, speaker_rep_dim]
        
        if uttr_reps != None:
            assert uttr_reps.shape[0] and uttr_bow.shape[0] == len(lens)
            assert uttr_reps.shape[-1] == self.speaker_rep_dim
        bs = uttr_bow.shape[0]
        speaker_idx =[[torch.arange(i,l,self.speaker_num) for i in range(self.speaker_num)] for l in lens]
        speaker_uttr_num = [[idx[i].shape[0] for idx in speaker_idx] for i in range(self.speaker_num)]
        max_speaker_uttr_num = [max(speaker_uttr_num[i]) for i in range(self.speaker_num)]

        speaker_uttr_bow = torch.zeros(bs*self.speaker_num, max(max_speaker_uttr_num), self.input_dim, requires_grad=False).to(self.device)
        speaker_uttr_reps = torch.zeros(bs*self.speaker_num, max(max_speaker_uttr_num), self.speaker_rep_dim, requires_grad=True).to(self.device)
        
        # print("self.input_dim", self.input_dim)
        # print("speaker_uttr_bow.shape", speaker_uttr_bow.shape)
        # print("speaker_uttr_reps.shape", speaker_uttr_reps.shape)

        for i in range(self.speaker_num):
            for b in range(bs):
                # print("speaker_uttr_bow[bs*i+b, :speaker_uttr_num[i][b], :].shape", speaker_uttr_bow[bs*i+b, :speaker_uttr_num[i][b], :].shape)
                # print("uttr_bow[b, speaker_idx[b][i], :].shape", uttr_bow[b, speaker_idx[b][i], :].shape)
                assert speaker_uttr_bow[bs*i+b, :speaker_uttr_num[i][b], :].shape == uttr_bow[b, speaker_idx[b][i], :].shape
                speaker_uttr_bow[bs*i+b, :speaker_uttr_num[i][b], :].copy_(uttr_bow[b, speaker_idx[b][i], :])
                if uttr_reps != None:
                    assert speaker_uttr_reps[bs*i+b, :speaker_uttr_num[i][b], :].shape ==  uttr_reps[b, speaker_idx[b][i], :].shape
                    speaker_uttr_reps[bs*i+b, :speaker_uttr_num[i][b], :].copy_(uttr_reps[b, speaker_idx[b][i], :])

        phi, mu, logvar = self.encode(speaker_uttr_bow, reps=speaker_uttr_reps, speaker_uttr_num=speaker_uttr_num)
        speaker_bow = F.normalize(speaker_uttr_bow, dim=-1, p=1)

        z = self.reparameterize(mu, logvar)  # theta
        theta = z.unsqueeze(1).expand(-1,phi.shape[1],-1)
        z = torch.tanh(self.global_to_local(torch.cat([phi, theta], dim=-1)))
        g = self.generate(z)

        recon_bow = self.decode(g)
        

        x_bow, x_rec = [], []
        if out_g:
            g_list = []
        for i in range(self.speaker_num):
            for b in range(bs):
                x_bow.append(speaker_bow[bs*i+b, :speaker_uttr_num[i][b], :])
                x_rec.append(recon_bow[bs*i+b, :speaker_uttr_num[i][b], :])
                if out_g:
                    g_list.append(g[bs*i+b, :speaker_uttr_num[i][b], :])
        x_bow = torch.cat(x_bow, dim=0)
        x_rec = torch.cat(x_rec, dim=0)
        if out_g:
            g_list = torch.cat(g_list, dim=0)

        if out_g:
            return g_list, x_bow, x_rec, mu, logvar
        else:
            return x_bow, x_rec, mu, logvar


    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta = self.get_beta().cpu().detach().numpy()
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        topic_words_all = {}
        for k, beta_k in enumerate(beta):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
            topic_words_all[k] = topic_words
        fw.close()
        return topic_words_all

    def eval_topic(self, feature_names, n_top_words=20, common_texts=None,candidate_wordid=None):
        beta = self.get_beta().cpu().detach().numpy()
        # print(beta, beta.shape)
        if candidate_wordid is not None:
            beta = beta[:,candidate_wordid]
            feature_names = [feature_names[i] for i in candidate_wordid]
        return self.print_top_words(beta, feature_names, n_top_words, common_texts)


    def print_top_words(self, beta, feature_names, n_top_words=20, common_texts=None):
        print('---------------Printing the Topics------------------')
        topic_set = []
        top_words = []
        beta_idx = np.argsort(beta,axis=-1)[:,::-1]
        for i in range(len(beta)):
            line = " ".join([feature_names[j]
                            for j in beta_idx[i, :n_top_words]])
            topic_set.append(line.split(' '))
            top_words.append([feature_names[j] for j in beta_idx[i, :50]])
            print('topic{}\t{}'.format(i, line))
        print('---------------End of Topics------------------')
        if common_texts:
            texts = [str(text).replace(';', '').split(' ') for text in common_texts]
            avg_td, avg_cv, avg_npmi = [], [], []
            for top_n in [5, 10, 15, 20, 25]:
                TD = TopicDiversity(topk=top_n).score(top_words=top_words)
                print('top{}, TD:{}'.format(top_n, TD))
                cv = Coherence(topk=top_n, texts=texts, measure='c_v').score(top_words=top_words)
                npmi = Coherence(topk=top_n, texts=texts, measure='c_npmi').score(top_words=top_words)
                uci = Coherence(topk=top_n, texts=texts, measure='c_uci').score(top_words=top_words)
                u_mass = Coherence(topk=top_n, texts=texts, measure='u_mass').score(top_words=top_words)
                print('top{}, cv: {:.6f}, npmi: {:.6f}, uci: {:.6f}, u_mass: {:.6f}'.format(top_n, cv, npmi, uci, u_mass))

                avg_td.append(TD)
                avg_cv.append(cv)
                avg_npmi.append(npmi)

            return avg_td, avg_cv, avg_npmi


    def get_topic_diversity(self, beta, n_top_words):
        num_topics = beta.shape[0]
        list_w = np.zeros((num_topics, n_top_words))
        for k in range(num_topics):
            idx = beta[k, :].argsort()[-n_top_words:][::-1]
            list_w[k, :] = idx
        n_unique = len(np.unique(list_w))
        TD = n_unique / (n_top_words * num_topics)
        print('Topic diveristy is: {} in top {} words'.format(TD, n_top_words))

    def mimno_topic_coherence(self, topic_words, texts, topn):
        tword_set = set([w for wlst in topic_words for w in wlst])
        word2docs = {w:set([]) for w in tword_set}
        for docid,doc in enumerate(texts):
            doc = set(doc)
            for word in tword_set:
                if word in doc:
                    word2docs[word].add(docid)
        def co_occur(w1,w2):
            return len(word2docs[w1].intersection(word2docs[w2]))+1
        scores = []
        for wlst in topic_words:
            s = 0
            for i in range(1,len(wlst[:topn])):
                for j in range(0,i):
                    s += np.log((co_occur(wlst[i],wlst[j])+1.0)/(len(word2docs[wlst[j]])+ 1e-6) )
            scores.append(s)
        return np.mean(s)

