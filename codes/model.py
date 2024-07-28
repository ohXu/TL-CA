from codes.positionEnc import *
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
RADIUS = 7


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False):
        # print(x.shape) torch.Size([16, 226, 32])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # print(qkv[0].shape, self.to_qkv(x).shape) torch.Size([16, 226, 96]) torch.Size([16, 226, 288])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # print(q.shape) torch.Size([16, 6, 226, 16])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print(dots.shape) torch.Size([16, 6, 226, 226])
        attn = self.attend(dots)
        attn = self.dropout(attn)
        if return_attention == True:
            return attn

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def getLastAttention2(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            # print(i, attn)
            if i != len(self.layers) - 1:
                # if i != 0:
                x = attn(x) + x
                x = ff(x) + x
            else:
                x = attn(x, return_attention=True)
                return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_encoder = GridCellSpatialRelationEncoder(spa_embed_dim=dim, frequency_num=6, ffn=True)
        self.pos_embedding1 = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = img[:, :-2, :, :]
        c = img[:, -2:, :, :]
        c = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)(c)
        c = c.detach().cpu().numpy()
        pos = self.pos_encoder(c)
        x = self.to_patch_embedding(x)
        # print(x.shape) torch.Size([16, 225, 32])
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape) torch.Size([16, 226, 32])

        pos_embeddings = repeat(self.pos_embedding1, '1 1 d -> b 1 d', b=b)
        pos_embeddings = torch.cat((pos_embeddings, pos), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x += pos_embeddings
        x = self.dropout(x)
        x = self.transformer(x)
        # print(x.shape) torch.Size([16, 226, 32])
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # print(x.shape) torch.Size([16, 32])
        x = self.to_latent(x)
        return self.mlp_head(x)

    def getLastAttention(self, img):
        x = img[:, :-2, :, :]
        c = img[:, -2:, :, :]
        c = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)(c)
        c = c.detach().cpu().numpy()
        pos = self.pos_encoder(c)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embeddings = repeat(self.pos_embedding1, '1 1 d -> b 1 d', b=b)
        pos_embeddings = torch.cat((pos_embeddings, pos), dim=1)

        x += pos_embeddings
        x = self.dropout(x)
        x = self.transformer.getLastAttention2(x)
        return x


class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight  # 二分类中正负样本的权重，第一项为负类权重，第二项为正类权重

    def forward(self, input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = - self.weight[1] * target * torch.log(input) - (1 - target) * self.weight[0] * torch.log(1 - input)
        return torch.mean(bce)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def func1(x):
    if x == 0:
        return x
    else:
        return 1 / x


def to_adj(coords, n):
    x = coords[:, 0].cpu()
    y = coords[:, 1].cpu()
    x = x.unsqueeze(1)
    x = x.repeat(1, n)
    y = y.unsqueeze(1)
    y = y.repeat(1, n)
    x_T = x.transpose(0, 1)
    y_T = y.transpose(0, 1)
    adj_matrix = torch.sqrt(torch.pow((x * 105 - x_T * 105), 2) + torch.pow((y * 89 - y_T * 89), 2))
    adj_matrix1 = torch.sqrt(torch.pow((x * 105 - x_T * 105), 2) + torch.pow((y * 89 - y_T * 89), 2))
    non_zero_indices = (adj_matrix != 0)
    adj_matrix = torch.where(non_zero_indices, 1 / adj_matrix, torch.tensor(0.0))
    adj_matrix = nn.functional.normalize(adj_matrix, p=2, dim=1)

    return adj_matrix, adj_matrix1


def knn_to_adj(knn, n):
    adj_matrix = torch.zeros(n, n)
    for i in range(len(knn[0])):
        tow = knn[0][i]
        fro = knn[1][i]
        adj_matrix[tow, fro] = 1

    adj_matrix_T = adj_matrix.T
    for i in range(adj_matrix_T.shape[0]):
        row_sum = torch.sum(adj_matrix_T[i, :])
        adj_matrix_T[i, :] = adj_matrix_T[i, :] / row_sum

    return adj_matrix_T


def normal_torch(tensor, min_val=0):
    t_min = torch.min(tensor)
    t_max = torch.max(tensor)
    if t_min == 0 and t_max == 0:
        return tensor
    if min_val == -1:
        tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
    if min_val == 0:
        tensor_norm = ((tensor - t_min) / (t_max - t_min))
    return tensor_norm


def lw_tensor_local_moran2(y, w_sparse, na_to_zero=True):
    n_1 = y.shape[0]
    z = y - y.mean()
    sy = y.std()
    z /= sy
    den = (z * z).sum()
    zl = w_sparse * z
    zl2 = w_sparse @ z
    mi2 = n_1 * z * zl / den
    mi = n_1 * z.reshape((z.shape[0], 1)) * zl / den

    if na_to_zero == True:
        mi[torch.isnan(mi)] = 0
    return mi


def train(dataloader, model, loss_fn, loss_fn2, optimizer, device, knn_number, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    max_V = -1
    min_V = 1000000000
    for batch, (X, y) in enumerate(dataloader):

        # time_start = time.time()
        X, y = X.to(device), y.to(device)
        coords = X[:, -2:, RADIUS, RADIUS]
        batch_size = X.shape[0]
        # moran_weight_matrix = knn_to_adj(knn_graph(coords, k=knn_number), batch_size).to(device).float()

        moran_weight_matrix, m2 = to_adj(coords, batch_size)
        moran_weight_matrix = moran_weight_matrix.to(device).float()
        m2 = m2.numpy()
        y2 = lw_tensor_local_moran2(y, moran_weight_matrix)

        # Compute prediction error
        pred = model(X)
        pred = pred.squeeze(-1)
        pred2 = lw_tensor_local_moran2(pred, moran_weight_matrix)

        loss = 0.5 * loss_fn(pred, y) + 0.5 * loss_fn2(pred2, y2)
        correct += sum(row.all().int().item() for row in (pred.ge(0.5) == y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Train_Time:", time.time() - time_start)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    print(max_V, min_V)


def val(dataloader, model, loss_fn, loss_fn2, device, knn_number, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            coords = X[:, -2:, RADIUS, RADIUS]
            batch_size = X.shape[0]

            moran_weight_matrix, m2 = to_adj(coords, batch_size)
            moran_weight_matrix = moran_weight_matrix.to(device).float()
            m2 = m2.numpy()
            y2 = lw_tensor_local_moran2(y, moran_weight_matrix)

            pred = model(X)
            pred = pred.squeeze(-1)
            pred2 = lw_tensor_local_moran2(pred, moran_weight_matrix)

            test_loss += 0.5 * loss_fn(pred, y).item() + 0.5 * loss_fn2(pred2, y2).item()
            correct += sum(row.all().int().item() for row in (pred.ge(0.5) == y))

    test_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    ans = []
    ans1 = []
    model.eval()
    test_loss, correct = 0, 0
    n = 0
    with torch.no_grad():
        for X, y in dataloader:
            n += 1
            # time_start = time.time()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(-1)
            test_loss += loss_fn(pred, y).item()
            correct += sum(row.all().int().item() for row in (pred.ge(0.5) == y))
            for i in range(y.shape[0]):
                ans.append(pred[i].item())
                ans1.append(y[i].item())
            # print("testTime:", time.time() - time_start)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return ans, ans1
