import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import requests
import torch

def generate_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def generate_conceptnet_related(word, rel="RelatedTo", limit=10):
    word = word.replace(" ", "_").lower()
    url = f"http://api.conceptnet.io/query?start=/c/en/{word}&rel=/r/{rel}&limit={limit}"

    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print("ConceptNet error:", e)
        return []

    related = set()
    for edge in data.get("edges", []):
        term = edge["end"]["label"]
        if term.lower() != word:
            related.add(term.lower())
    return list(related)

def filter_clip_synonyms(affordance, candidate_prompts, clip_model, tokenizer, device, best_k=3, synonymsOnly=True):
    filtered_candidates = [c for c in candidate_prompts if c.lower() != affordance]
    if len(filtered_candidates) == 0:
      return []

    all_prompts = [affordance] + candidate_prompts
    text_tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    base_feature = text_features[0].unsqueeze(0)
    candidate_features = text_features[1:]

    similarity = torch.cosine_similarity(base_feature, candidate_features, dim=-1)
    best_indices = []
    for i in range(best_k, 0, -1):
      try:
        k = 0
        if synonymsOnly:
          k = best_k + 1
        else:
          k = best_k

        best_indices = torch.topk(similarity, k=k).indices
        break
      except:
        continue

    if len(best_indices) == 0:
      return []
    best_synonyms = [candidate_prompts[i] for i in best_indices]
    best_synonyms = list(set(best_synonyms))
    best_synonyms = best_synonyms[1:]

    return best_synonyms

def generate_prompts(prompt_template, semantic_class, affordance, clip_model, tokenizer, device, weight=0.55):
    synonyms = generate_synonyms(affordance)
    synonyms = filter_clip_synonyms(affordance, synonyms, clip_model, tokenizer, device)

    related_words = generate_conceptnet_related(affordance)
    related_words = related_words + generate_conceptnet_related(affordance, "AtLocation")
    related_words = related_words + generate_conceptnet_related(affordance, "LocatedNear")
    related_words = related_words + generate_conceptnet_related(affordance, "IsA")
    related_words = filter_clip_synonyms(affordance, related_words, clip_model, tokenizer, device)

    words = [affordance] + synonyms + related_words
    prompts_list = [prompt_template.format(semantic_class, word) for word in words]

    num_synonyms = len(synonyms + related_words)
    remaining_weight = 1.0 - weight
    synonym_weight = remaining_weight / num_synonyms if num_synonyms > 0 else 0.0
    weights_list = [weight] + [synonym_weight] * num_synonyms
    weights_list = [round(w, 3) for w in weights_list]

    return prompts_list, weights_list