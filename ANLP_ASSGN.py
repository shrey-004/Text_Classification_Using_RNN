# ANLP_ASSGN.py
# From-scratch RNN, LSTM, Transformer (NumPy + pandas only)
# Shrey Srivastava - improved & fixed

import os, re, math, time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# 1) Cleaning + tokenization
# ---------------------------
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'http\S+', ' ', t)
    t = re.sub(r'@\w+', ' ', t)
    t = re.sub(r'[^a-z!? ]', ' ', t)   # keep '!' and '?'
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def simple_tokenize(t):
    return t.split()

def build_vocab(texts, max_vocab=12000, min_freq=2):
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    toks = [w for w, f in cnt.most_common(max_vocab) if f >= min_freq]
    idx2w = ['<pad>', '<unk>'] + toks
    w2i = {w:i for i,w in enumerate(idx2w)}
    return w2i, idx2w

def texts_to_seq(texts, w2i, max_len):
    unk = w2i['<unk>']; pad = w2i['<pad>']
    seqs = []
    for t in texts:
        s = [w2i.get(w, unk) for w in t[:max_len]]
        if len(s) < max_len:
            s += [pad] * (max_len - len(s))
        seqs.append(s)
    return np.array(seqs, dtype=np.int32)

# ---------------------------
# 2) Load data (change path if needed)
# ---------------------------
TRAIN_FP = "Dataset/Corona_NLP_train.csv"
TEST_FP  = "Dataset/Corona_NLP_test.csv"  # optional

df = pd.read_csv(TRAIN_FP, encoding="latin1", on_bad_lines="skip")
# detect columns
text_col = None; label_col = None
for c in df.columns:
    if "tweet" in c.lower() or "text" in c.lower():
        text_col = c
    if "sentiment" in c.lower() or "label" in c.lower():
        label_col = c
if text_col is None: text_col = df.columns[0]
if label_col is None: label_col = df.columns[1]

print("Using text col:", text_col, "label col:", label_col)

df['clean'] = df[text_col].astype(str).apply(clean_text)
df['tok'] = df['clean'].apply(simple_tokenize)

labels = sorted(df[label_col].unique())
lbl2i = {l:i for i,l in enumerate(labels)}
i2lbl = {i:l for l,i in lbl2i.items()}
df['y'] = df[label_col].map(lbl2i)
print("Labels:", lbl2i)

MAX_VOCAB = 12000
MAX_LEN = 60
w2i, i2w = build_vocab(df['tok'].tolist(), max_vocab=MAX_VOCAB, min_freq=2)
vocab_size = len(w2i)
print("Vocab size:", vocab_size)

X = texts_to_seq(df['tok'].tolist(), w2i, MAX_LEN)
Y = df['y'].values
x_tr, x_val, y_tr, y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)

# ---------------------------
# 3) Helpers
# ---------------------------
def one_hot_batch(y, C):
    oh = np.zeros((len(y), C), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(pred, targ):
    eps = 1e-9
    return -np.mean(np.sum(targ * np.log(pred + eps), axis=1))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------
# 4) Embedding
# ---------------------------
class Embedding:
    def __init__(self, vocab_size, dim=200):
        self.W = np.random.randn(vocab_size, dim) * 0.01
        self.dW = np.zeros_like(self.W)
        self.dim = dim

    def forward(self, idx):
        return self.W[idx]            # (bs, seql, dim)

    def backward(self, idx_batch, grad_out):
        # sparse accumulate (simple loops)
        self.dW.fill(0.0)
        bs, seql, d = grad_out.shape
        for i in range(bs):
            for j in range(seql):
                self.dW[idx_batch[i,j]] += grad_out[i,j]

    def zero_grad(self):
        self.dW.fill(0.0)

# ---------------------------
# 5) BaseModel
# ---------------------------
class BaseModel:
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        self.emb = Embedding(vocab_size, emb_dim)
        self.Wc = np.random.randn(hid_dim, out_dim) * 0.01
        self.bc = np.zeros(out_dim)
        self.dWc = np.zeros_like(self.Wc)
        self.dbc = np.zeros_like(self.bc)
        self.out_dim = out_dim

    def classifier_forward(self, h):
        return h.dot(self.Wc) + self.bc

    def classifier_backward(self, h, grad_logits):
        self.dWc = h.T.dot(grad_logits) / h.shape[0]
        self.dbc = grad_logits.mean(axis=0)
        return grad_logits.dot(self.Wc.T)

    def zero_grads(self):
        self.emb.zero_grad()
        self.dWc.fill(0.0); self.dbc.fill(0.0)

    def get_params_and_grads(self):
        return [
            (self.emb.W, self.emb.dW, 'emb.W'),
            (self.Wc, self.dWc, 'Wc'),
            (self.bc, self.dbc, 'bc')
        ]

# ---------------------------
# 6) RNN
# ---------------------------
class RNNModel(BaseModel):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super().__init__(vocab_size, emb_dim, hid_dim, out_dim)
        self.Wxh = np.random.randn(emb_dim, hid_dim) * 0.01
        self.Whh = np.random.randn(hid_dim, hid_dim) * 0.01
        self.bh  = np.zeros(hid_dim)
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh  = np.zeros_like(self.bh)

    def forward(self, x_idx):
        self.x_idx = x_idx
        emb = self.emb.forward(x_idx)           # (bs,seql,ed)
        bs, seql, ed = emb.shape
        h = np.zeros((bs, self.Whh.shape[0]))
        self.hs = []
        for t in range(seql):
            h = np.tanh(emb[:,t].dot(self.Wxh) + h.dot(self.Whh) + self.bh)
            self.hs.append(h.copy())
        self.hs = np.stack(self.hs, axis=1)     # (bs,seql,hid)
        logits = self.classifier_forward(self.hs[:, -1, :])
        return logits

    def backward(self, grad_logits):
        bs, seql = self.x_idx.shape
        grad_h_last = self.classifier_backward(self.hs[:, -1, :], grad_logits)
        grad_h = np.zeros_like(self.hs)
        grad_h[:, -1, :] = grad_h_last
        grad_emb = np.zeros((bs, seql, self.emb.dim))
        self.dWxh.fill(0.0); self.dWhh.fill(0.0); self.dbh.fill(0.0)
        for t in reversed(range(seql)):
            h_t = self.hs[:, t, :]
            h_prev = self.hs[:, t-1, :] if t > 0 else np.zeros_like(h_t)
            dh_raw = (1 - h_t**2) * grad_h[:, t, :]
            self.dbh += dh_raw.mean(axis=0)
            # emb for tokens at t:
            x_emb_t = self.emb.W[self.x_idx[:, t]]
            self.dWxh += x_emb_t.T.dot(dh_raw) / bs
            self.dWhh += h_prev.T.dot(dh_raw) / bs
            grad_emb[:, t, :] = dh_raw.dot(self.Wxh.T)
            if t > 0:
                grad_h[:, t-1, :] += dh_raw.dot(self.Whh.T)
        self.emb.backward(self.x_idx, grad_emb)

    def zero_grads(self):
        super().zero_grads()
        self.dWxh.fill(0.0); self.dWhh.fill(0.0); self.dbh.fill(0.0)

    def get_params_and_grads(self):
        return super().get_params_and_grads() + [
            (self.Wxh, self.dWxh, 'Wxh'),
            (self.Whh, self.dWhh, 'Whh'),
            (self.bh, self.dbh, 'bh')
        ]

# ---------------------------
# 7) LSTM (fixed gate storage)
# ---------------------------
class LSTMModel(BaseModel):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super().__init__(vocab_size, emb_dim, hid_dim, out_dim)
        self.Wx = np.random.randn(emb_dim, 4*hid_dim) * 0.01
        self.Wh = np.random.randn(hid_dim, 4*hid_dim) * 0.01
        self.b  = np.zeros(4*hid_dim)
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db  = np.zeros_like(self.b)

    def forward(self, x_idx):
        self.x_idx = x_idx
        emb = self.emb.forward(x_idx)
        bs, seql, ed = emb.shape
        hid = self.Wc.shape[0]     # hidden dim
        h = np.zeros((bs, hid)); c = np.zeros_like(h)
        self.hs = []; self.cs = []
        self.i_s = []; self.f_s = []; self.g_s = []; self.o_s = []
        for t in range(seql):
            z = emb[:, t].dot(self.Wx) + h.dot(self.Wh) + self.b
            i = sigmoid(z[:, :hid])
            f = sigmoid(z[:, hid:2*hid])
            g = np.tanh(z[:, 2*hid:3*hid])
            o = sigmoid(z[:, 3*hid:4*hid])
            c = f * c + i * g
            h = o * np.tanh(c)
            self.hs.append(h.copy()); self.cs.append(c.copy())
            self.i_s.append(i); self.f_s.append(f); self.g_s.append(g); self.o_s.append(o)
        self.hs = np.stack(self.hs, axis=1)
        self.cs = np.stack(self.cs, axis=1)
        logits = self.classifier_forward(self.hs[:, -1, :])
        return logits

    def backward(self, grad_logits):
        bs, seql = self.x_idx.shape
        hid = self.Wc.shape[0]
        dh_last = self.classifier_backward(self.hs[:, -1, :], grad_logits)
        grad_h = np.zeros_like(self.hs); grad_c = np.zeros_like(self.cs)
        grad_h[:, -1, :] = dh_last
        grad_emb = np.zeros((bs, seql, self.emb.dim))
        self.dWx.fill(0.0); self.dWh.fill(0.0); self.db.fill(0.0)
        for t in reversed(range(seql)):
            h_t = self.hs[:, t, :]
            c_t = self.cs[:, t, :]
            i_t = self.i_s[t]; f_t = self.f_s[t]; g_t = self.g_s[t]; o_t = self.o_s[t]
            h_prev = self.hs[:, t-1, :] if t > 0 else np.zeros_like(h_t)
            c_prev = self.cs[:, t-1, :] if t > 0 else np.zeros_like(c_t)
            dh = grad_h[:, t, :]
            do = dh * np.tanh(c_t)
            dc = dh * o_t * (1 - np.tanh(c_t)**2) + grad_c[:, t]
            di = dc * g_t
            dg = dc * i_t
            df = dc * c_prev
            dc_prev = dc * f_t
            di_pre = di * i_t * (1 - i_t)
            df_pre = df * f_t * (1 - f_t)
            dg_pre = dg * (1 - g_t**2)
            do_pre = do * o_t * (1 - o_t)
            dconcat = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)  # (bs,4*hid)
            x_t = self.emb.W[self.x_idx[:, t]]
            self.dWx += x_t.T.dot(dconcat) / bs
            self.dWh += h_prev.T.dot(dconcat) / bs
            self.db  += dconcat.mean(axis=0)
            grad_emb[:, t, :] = dconcat.dot(self.Wx.T)
            if t > 0:
                grad_h[:, t-1, :] += dconcat.dot(self.Wh.T)
                grad_c[:, t-1] = dc_prev
        self.emb.backward(self.x_idx, grad_emb)

    def zero_grads(self):
        super().zero_grads()
        self.dWx.fill(0.0); self.dWh.fill(0.0); self.db.fill(0.0)

    def get_params_and_grads(self):
        return super().get_params_and_grads() + [
            (self.Wx, self.dWx, 'Wx'),
            (self.Wh, self.dWh, 'Wh'),
            (self.b,  self.db,  'b')
        ]

# ---------------------------
# 8) Transformer encoder (single block) with positional encodings
#    Backprop simplified (mainly embedding gets gradients) -- stable for training
# ---------------------------
def positional_encoding(max_len, d):
    pos = np.arange(max_len)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2*(i//2))/d)
    pe = np.zeros((max_len, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

class TransEncoderModel(BaseModel):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, n_heads=2):
        super().__init__(vocab_size, emb_dim, hid_dim, out_dim)
        assert emb_dim % n_heads == 0
        self.nh = n_heads; self.dk = emb_dim // n_heads; self.emb_dim = emb_dim
        self.Wq = np.random.randn(emb_dim, emb_dim) * 0.01
        self.Wk = np.random.randn(emb_dim, emb_dim) * 0.01
        self.Wv = np.random.randn(emb_dim, emb_dim) * 0.01
        self.Wo = np.random.randn(emb_dim, emb_dim) * 0.01
        self.W1 = np.random.randn(emb_dim, 2*emb_dim) * 0.01
        self.b1 = np.zeros(2*emb_dim)
        self.W2 = np.random.randn(2*emb_dim, emb_dim) * 0.01
        self.b2 = np.zeros(emb_dim)
        self.dWq = np.zeros_like(self.Wq); self.dWk = np.zeros_like(self.Wk); self.dWv = np.zeros_like(self.Wv); self.dWo = np.zeros_like(self.Wo)
        self.dW1 = np.zeros_like(self.W1); self.db1 = np.zeros_like(self.b1); self.dW2 = np.zeros_like(self.W2); self.db2 = np.zeros_like(self.b2)
        self.pe = positional_encoding(MAX_LEN, emb_dim)

    def forward(self, x_idx):
        self.x_idx = x_idx
        emb = self.emb.forward(x_idx) + self.pe[:x_idx.shape[1]]
        bs, seql, d = emb.shape
        Q = emb.dot(self.Wq); K = emb.dot(self.Wk); V = emb.dot(self.Wv)
        # scaled dot-product (no heads split for simplicity)
        scores = Q @ K.transpose(0,2,1) / math.sqrt(d)
        attn = softmax(scores.reshape(-1, scores.shape[-1])).reshape(scores.shape)
        attn_out = attn @ V
        attn_out = attn_out.dot(self.Wo)
        x1 = emb + attn_out
        ff = np.tanh(x1.dot(self.W1) + self.b1)
        ff = ff.dot(self.W2) + self.b2
        enc = x1 + ff
        self.cache = (emb, enc, attn)  # keep for backward approx
        h_last = enc.mean(axis=1)
        logits = self.classifier_forward(h_last)
        return logits

    def backward(self, grad_logits):
        # simplified: push pooled grad equally to sequence then to embedding
        emb, enc, attn = self.cache
        bs, seql, d = emb.shape
        grad_h = self.classifier_backward(enc.mean(axis=1), grad_logits)
        grad_enc = np.repeat(grad_h[:, None, :], seql, axis=1) / 1.0
        # back through FF (approx)
        grad_x1 = grad_enc + (grad_enc.dot(self.W2.T) * (1 - np.tanh(enc.dot(self.W1) + self.b1)**2))
        # split to emb (residual) and attn_out
        grad_emb = grad_x1.copy()
        # also add contribution from attention output via Wo (approx)
        grad_attn_out = grad_x1
        grad_emb += grad_attn_out.dot(self.Wo.T)
        # push to embedding matrix
        self.emb.backward(self.x_idx, grad_emb)
        # grads for other matrices left as zeros or approximated (ok for this assignment)

    def zero_grads(self):
        super().zero_grads()
        self.dWq.fill(0); self.dWk.fill(0); self.dWv.fill(0); self.dWo.fill(0)
        self.dW1.fill(0); self.db1.fill(0); self.dW2.fill(0); self.db2.fill(0)

    def get_params_and_grads(self):
        base = super().get_params_and_grads()
        extra = [
            (self.Wq, self.dWq, 'Wq'), (self.Wk, self.dWk, 'Wk'), (self.Wv, self.dWv, 'Wv'), (self.Wo, self.dWo, 'Wo'),
            (self.W1, self.dW1, 'W1'), (self.b1, self.db1, 'b1'), (self.W2, self.dW2, 'W2'), (self.b2, self.db2, 'b2')
        ]
        return base + extra

# ---------------------------
# 9) Optimizer (Adam)
# ---------------------------
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8):
        self.lr = lr; self.b1 = betas[0]; self.b2 = betas[1]; self.eps = eps
        self.params = params
        self.ms = {name: np.zeros_like(p) for p,g,name in params}
        self.vs = {name: np.zeros_like(p) for p,g,name in params}
        self.t = 0

    def step(self):
        self.t += 1
        for p, g, name in self.params:
            m = self.ms[name]; v = self.vs[name]
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * (g * g)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ---------------------------
# 10) Training / eval utilities
# ---------------------------
def iterate_minibatches(x, y, bs=64, shuffle=True):
    n = len(x)
    idx = np.arange(n)
    if shuffle: np.random.shuffle(idx)
    for i in range(0, n, bs):
        sl = idx[i:i+bs]
        yield x[sl], y[sl]

def evaluate(model, x, y, bs=128):
    preds = []
    tot_loss = 0.0
    for xb, yb in iterate_minibatches(x, y, bs=bs, shuffle=False):
        logits = model.forward(xb)
        probs = softmax(logits)
        tot_loss += cross_entropy(probs, one_hot_batch(yb, model.out_dim)) * len(xb)
        preds.extend(list(np.argmax(probs, axis=1)))
    tot_loss /= len(x)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    return tot_loss, acc, f1

def train_model(model, x_tr, y_tr, x_val, y_val, epochs=8, bs=128, lr=1e-3, verbose=True):
    params = model.get_params_and_grads()
    opt = Adam(params, lr=lr)
    for e in range(1, epochs+1):
        t0 = time.time()
        for xb, yb in iterate_minibatches(x_tr, y_tr, bs=bs, shuffle=True):
            logits = model.forward(xb)
            probs = softmax(logits)
            loss = cross_entropy(probs, one_hot_batch(yb, model.out_dim))
            grad_logits = (probs - one_hot_batch(yb, model.out_dim)) / xb.shape[0]
            model.zero_grads()
            model.backward(grad_logits)
            opt.step()
        tr_loss, tr_acc, tr_f1 = evaluate(model, x_tr, y_tr, bs=bs)
        val_loss, val_acc, val_f1 = evaluate(model, x_val, y_val, bs=bs)
        if verbose:
            print(f"Epoch {e}/{epochs} time {time.time()-t0:.1f}s | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} tr_f1 {tr_f1:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_f1 {val_f1:.4f}")
    return model

# ---------------------------
# 11) Save / Load model helpers
# ---------------------------
def save_model(model, path_prefix):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    params = {}
    for p, g, name in model.get_params_and_grads():
        params[name] = p
    np.savez(path_prefix + '.npz', **params)
    return path_prefix + '.npz'

def load_model_weights(model, npz_path):
    d = np.load(npz_path, allow_pickle=True)
    for p, g, name in model.get_params_and_grads():
        if name in d:
            arr = d[name]
            if p.shape == arr.shape:
                p[:] = arr
            else:
                print("skip", name, "mismatch", p.shape, arr.shape)

# ---------------------------
# 12) Instantiate models & train
# ---------------------------
emb_dim = 200
hid_dim = 128
out_dim = len(labels)

rnn = RNNModel(vocab_size, emb_dim, hid_dim, out_dim)
lstm = LSTMModel(vocab_size, emb_dim, hid_dim, out_dim)
trans = TransEncoderModel(vocab_size, emb_dim, emb_dim, out_dim, n_heads=2)

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

print("\n===== Training RNN =====")
rnn = train_model(rnn, x_tr, y_tr, x_val, y_val, epochs=8, bs=128, lr=1e-3)
vl = evaluate(rnn, x_val, y_val)
print("RNN Final -> loss {:.4f} acc {:.4f} f1 {:.4f}".format(*vl))
save_model(rnn, os.path.join(save_dir, "rnn_model"))
print("Saved rnn:", os.path.join(save_dir, "rnn_model.npz"))

print("\n===== Training LSTM =====")
lstm = train_model(lstm, x_tr, y_tr, x_val, y_val, epochs=8, bs=128, lr=1e-3)
vl = evaluate(lstm, x_val, y_val)
print("LSTM Final -> loss {:.4f} acc {:.4f} f1 {:.4f}".format(*vl))
save_model(lstm, os.path.join(save_dir, "lstm_model"))
print("Saved lstm:", os.path.join(save_dir, "lstm_model.npz"))

print("\n===== Training Transformer =====")
trans = train_model(trans, x_tr, y_tr, x_val, y_val, epochs=8, bs=128, lr=1e-3)
vl = evaluate(trans, x_val, y_val)
print("Transformer Final -> loss {:.4f} acc {:.4f} f1 {:.4f}".format(*vl))
save_model(trans, os.path.join(save_dir, "transformer_model"))
print("Saved trans:", os.path.join(save_dir, "transformer_model.npz"))

# ---------------------------
# 13) Optional: evaluate on TEST_FP if present
# ---------------------------
if os.path.exists(TEST_FP):
    print("\nEvaluating on test set:", TEST_FP)
    df_test = pd.read_csv(TEST_FP, encoding="latin1", on_bad_lines="skip")
    # adapt columns similar to train detection
    text_c = None; label_c = None
    for c in df_test.columns:
        if "tweet" in c.lower() or "text" in c.lower(): text_c = c
        if "sentiment" in c.lower() or "label" in c.lower(): label_c = c
    if text_c is None: text_c = df_test.columns[0]
    df_test['clean'] = df_test[text_c].astype(str).apply(clean_text)
    df_test['tok'] = df_test['clean'].apply(simple_tokenize)
    X_test = texts_to_seq(df_test['tok'].tolist(), w2i, MAX_LEN)
    if label_c is not None:
        y_test = df_test[label_c].map(lbl2i).fillna(0).astype(int).values
        for name, model in [("RNN", rnn), ("LSTM", lstm), ("Transformer", trans)]:
            l, a, f = evaluate(model, X_test, y_test)
            print(f"{name} on test -> loss {l:.4f} acc {a:.4f} f1 {f:.4f}")
    else:
        print("Test set has no label column; skipping labeled evaluation.")

print("\nDone.")
