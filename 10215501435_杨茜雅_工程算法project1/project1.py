import random
import time
from collections import defaultdict

with open('ca-AstroPh.txt', 'r') as f:
    edges = [tuple(map(int, line.strip().split())) for line in f.readlines()]

neighbors = defaultdict(set)
for a, b in edges:
    neighbors[a].add(b)
    neighbors[b].add(a)

jaccard = {}
for b in neighbors.keys():
    if b!= 1:
        common_neighbors = neighbors[b] & neighbors[1]
        union_neighbors = neighbors[b]|neighbors[1]
        jaccard[b] = len(common_neighbors) / len(union_neighbors)

top_k = 10
most_similar = sorted(jaccard.items(), key=lambda x: x[1], reverse=True)[:top_k]
#输出结果
print(f"before searching")
for b, score in most_similar:
    print(f"Node{b}:jaccard similarity score = {score:.4f}")

#  构建邻接表
list = {}
for e in edges:
    if e[0] not in list:
        list[e[0]] = set()
    list[e[0]].add(e[1])
    if e[1] not in list:
        list[e[1]] = set()
    list[e[1]].add(e[0])


# MinHash哈希函数族
class MinHash:
    def __init__(self, n_perm):
        self.n_perm = n_perm
        self.permutations = []
        for i in range(n_perm):
            p = list(range(len(list)))
            random.shuffle(p)
            self.permutations.append(p)

    def hash(self, s):
        hashes = []
        for p in self.permutations:
            for v in s:
                if v in p:
                    hashes.append(p.index(v))
                    break
        return hashes


class LSH:
    def __init__(self, b, r, n_perm):
        self.b = b
        self.r = r
        self.n_perm = n_perm
        self.minhash = MinHash(n_perm)
        self.buckets = {}
        for i in range(b):
            self.buckets[i] = {}

    def index(self):
        start_time = time.time()
        for node in list:
            signature = self._signature(list[node])
            for i in range(self.b):
                bucket_key = tuple(signature[i * self.r:(i + 1) * self.r])
                if bucket_key not in self.buckets[i]:
                    self.buckets[i][bucket_key] = set()
                self.buckets[i][bucket_key].add(node)
        end_time = time.time()
        query_time = end_time - start_time

    def query(self, query_node, k):
        start_time = time.time()
        query_set = list[query_node]
        query_signature = self._signature(query_set)
        candidate_set = set()
        for i in range(self.b):
            bucket_key = tuple(query_signature[i * self.r:(i + 1) * self.r])
            if bucket_key in self.buckets[i]:
                candidate_set.update(self.buckets[i][bucket_key])
            candidate_set.discard(query_node)
            candidates = [(node, self._jaccard_similarity(query_set, list[node])) for node in candidate_set]
            candidates.sort(key=lambda x: -x[1])
        end_time = time.time()
        return candidates[:k]

    def _signature(self, s):
        return self.minhash.hash(s)

    def _jaccard_similarity(self, s1, s2):
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        return len(s1 & s2) / len(s1 | s2)


lsh = LSH(b=10, r=10, n_perm=10)
lsh.index()


query_node = 1
k = 10
result = lsh.query(query_node, k)
print(f"Top {k} nodes most similar to node {query_node}:")
for i, (node, similarity) in enumerate(result):
    print(f"{i + 1}. Node {node}: Jaccard similarity score = {similarity:.3f}")

# 空间使用率
total_size = 0
for i in range(lsh.b):
    total_size += sum(len(s) for s in lsh.buckets[i].values())
space_usage = total_size / len(list)
print(f"Space usage: {space_usage:.4f} average nodes per bucket")

# 评估搜索准确性
total_jaccard_sim = 0.0
for query_node in list.keys():
    result = lsh.query(query_node, k)
    true_neighbors = list[query_node]
    jaccard_sum = sum(similarity for _, similarity in result if _ in true_neighbors)
    total_jaccard_sim += jaccard_sum / len(true_neighbors)
    avg_jaccard_sim = total_jaccard_sim / len(list)
print(f"Jaccard similarity score: {avg_jaccard_sim:.4f}")

# 评估搜索时间
total_query_time = 0.0
for query_node in list.keys():
    start_time = time.time()
    lsh.query(query_node, k)
    query_time = time.time() - start_time
    total_query_time += query_time
    avg_query_time = total_query_time / len(list)
print(f"query time: {avg_query_time:.4f} s")

# 评估索引时间
total_index_time = 0.0
for query_node in list.keys():
    start_time = time.time()
    lsh.index()
    index_time = time.time() - start_time
    total_index_time += index_time
    avg_index_time = total_index_time / len(list)
print(f"Index time: {avg_index_time:.4f} s")
