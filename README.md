# Embedding-based LLM Alignment: 
## A Minimalist, Efficient, and Effective Infrastructure of Reward Modeling Research.
Codebase for Paper _[Reusing Embeddings: Reproducible Reward Model Research in Large Language Model Alignment without GPUs](https://arxiv.org/pdf/2502.04357)_


-----
### 🚀 Example Usage

```python3
# Specify Task, Embedding Model, Response Generation Model
args.task = 'Harmless'
args.res_gen_model = 'Gemma2b-sft'
args.embed_model = 'Gemma2b'

# Load Training Data
train_embeddings, train_rewards = load_embd_data(task=args.task, res_gen_model=args.res_gen_model, embed_model=args.embed_model, split='train') 
### train_embeddings.shape = (40000, 10, 2048), 40000 prompts, 10 responses for each prompt, Gemma2b has a 2048-dim embedding space
### train_rewards.shape = (40000, 10, 1), corresponding reward

# Load Testing Data
test_embeddings, test_rewards = load_embd_data(task=args.task, res_gen_model=args.res_gen_model, embed_model=args.embed_model, split='test')
### test_embeddings.shape = (2000, 500, 2048)
### test_rewards.shape = (2000, 500, 1)

# Generation of Pairwise Comparisons
train_comparisons, train_labels = pair_annotate(train_embeddings, train_rewards, annotation_quality = 0.1)
# annotation noise can be adjusted through "annotation_quality"

# Train Embedding-based Reward Model (e.g., use a Bradley-Terry MLP)
reward_model = BT_MLP()
reward_model.fit(train_comparisons, train_labels)

# Make Predictions with the Reward Model on Testset
rm_predictions = reward_model.predict(test_embeddings)
print(rm_predictions.shape) 
### (2000, 500, 1)

# Calculate Evaluation Metrics on Testset
bon_500 = calc_bon(rm_predictions, test_rewards, N=500)
spearmanr = calc_spearmanr(rm_predictions, test_rewards)
```

----
### 🔨 Build (TBD)

```python3
pip install 
```

----
### 📊 Embedding Data Downloading
Here is a Google Drive link for **a single experiment setup**, which is about 10GB. It can be used for a quick start/reproduction:

[Google Drive Link (10GB)](https://drive.google.com/drive/folders/1Op0B1jc4Zr6t6DFWyLcCulpq67CJOYsU?usp=sharing)

The full 300GB embedding files can be found at:

[Google Drive Link (300GB)](https://drive.google.com/drive/folders/1cRiwvZDxlq_5DVHBIIVYjeunse42ALMO?usp=sharing)



---
### Demonstrative Use Cases

#### 1. A Quick Implementation of Reward Model Ensemble


#### 2. A Quick Implementation of Active Reward Modeling 


#### 3. A Quick Implementation of Classification-based Reward Models


#### 4. Exciting Future Works!


