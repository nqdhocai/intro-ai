from enviroment.embedding import EmbeddingModel
from enviroment.retriever import FaissRetriever
from enviroment.env import RAGTopKEnv

from processing_data import train_dataset, corpus 

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

# Define model IDs
MODEL_IDS = {
    "bert": "google-bert/bert-base-multilingual-uncased",
    "e5-base": "intfloat/multilingual-e5-base",
    "bge-m3": "AITeamVN/Vietnamese_Embedding",
    "minilm": "google-minilm/multilingual-e5-base",
    "e5-small": "intfloat/multilingual-e5-small"
}

def initialize_embedding_models(model_ids):
    """Initialize embedding models."""
    return {
        model_name: EmbeddingModel(model_id) 
        for model_name, model_id in model_ids.items()
    }

def initialize_retrievers(embedding_models):
    """Initialize retrievers."""
    return {
        model_name: FaissRetriever(embedding_model) 
        for model_name, embedding_model in embedding_models.items()
    }

def build_indices(retrievers, docs, cids):
    """Build indices for retrievers."""
    for model_name, retriever in retrievers.items():
        retriever.build_index(docs, cids)

def initialize_envs(retrievers, dataset, k_candidates, max_docs):
    """Initialize environments."""
    return {
        model_name: RAGTopKEnv(
            dataset=dataset,
            retriever=retriever,
            k_candidates=k_candidates,
            max_docs=max_docs
        ) for model_name, retriever in retrievers.items()
    }

def initialize_agents(agent_class, envs):
    """Initialize agents."""
    return {
        model_name: agent_class(
            state_dim=env.state_dim,
            action_dim=env.action_space.n
        ) for model_name, env in envs.items()
    }

embedding_models = initialize_embedding_models(MODEL_IDS)
retrievers = initialize_retrievers(embedding_models)

docs = corpus["truncated_text"].to_list()
cids = corpus["cid"].to_list()
build_indices(retrievers, docs, cids)

envs = initialize_envs(
    retrievers=retrievers,
    dataset=train_dataset,
    k_candidates=[0, 1, 3, 5, 7, 9],
    max_docs=10
)

dqn_agents = initialize_agents(DQNAgent, envs)
ppo_agents = initialize_agents(PPOAgent, envs)