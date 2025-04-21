# Load Training Data
train_embeddings, train_rewards = load_embd_data(task='Harmless', split='train')
print(train_embeddings.shape) 
### (40000, 10, 2048)
print(train_rewards.shape) 
### (40000, 10, 1)

# Load Testing Data
test_embeddings, test_rewards = load_embd_data(task='Harmless', split='test')
print(test_embeddings.shape) 
### (2000, 500, 2048)
print(test_rewards.shape) 
### (2000, 500, 1)

# Generation of Pairwise Comparisons
train_comparisons, train_labels = pair_annotate(train_embeddings, train_rewards)

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
