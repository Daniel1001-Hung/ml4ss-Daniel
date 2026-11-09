# 02 Explanation · STATS 201 Week 3

**Executive summary.** How could a researcher find recurring themes in a collection of public comments or creative requests? This lab turns a small WildChat-derived collection into numerical features and groups requests by similar word use. You learn how the text sample and its representation shape the patterns, how to read cluster sizes and center terms, and how to check a tentative theme when the number of groups changes.

[Open 02 Explanation in Colab](https://colab.research.google.com/github/sunshineluyao/ml4ss/blob/main/modules/02-explanation/week03_02_unsupervised_explanation.ipynb)

## Learning objectives

- Explain how a text request becomes a row of word weights.
- Interpret K-means groups using their sizes and prominent terms.
- Distinguish the information in a two-dimensional display from the features used to form the groups.

## Terms to know

| Term | Meaning in this lab |
|---|---|
| Feature | A numerical weight for one word in a request. |
| TF–IDF | Weights word use within a request against how common the word is across the collection. |
| `K` and center | `K` is the chosen number of groups. Each center is the average feature vector of its assigned requests. |
| Projection | A compressed display of the feature vectors. Truncated SVD creates the two coordinates used for the scatterplot. |

## Code cell walkthrough

### Cell 1 · Prepare the text tools (`setup`)

A librarian might look for related topics across many requests before choosing a set of catalogue tags. The setup brings in `TfidfVectorizer` to represent the text, `KMeans` to find groups, and `TruncatedSVD` to display the result. It also defines the public data request and plotting helpers. These imports let later cells separate the decisions about the sample, the features and the display.

<details>
<summary>Code from this cell</summary>

```python
# Read and run. The data loader makes public Hugging Face requests.

import json
import time
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as NotebookImage, display

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

DATA_SERVER = "https://datasets-server.huggingface.co"
WILDCHAT = "heyyjudes/wildchat-creative-only-labeled"

def read_url(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=25) as response:
                return response.read()
        except (URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"Hugging Face returned HTTP {error.code}. Rerun the data cell.") from error
            if attempt == 2:
                raise RuntimeError("Hugging Face is unavailable. The saved default outputs can support discussion.") from error
            time.sleep(2 ** attempt)

def page(dataset, offset=0, first=False):
    params = {"dataset": dataset, "config": "default", "split": "train"}
    endpoint = "first-rows" if first else "rows"
    if not first:
        params.update(offset=offset, length=100)
    return json.loads(read_url(f"{DATA_SERVER}/{endpoint}?{urlencode(params)}"))["rows"]

def show_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    display(NotebookImage(data=buffer.getvalue()))
```

</details>

### Cell 2 · Decide which requests enter the analysis (`cluster-data`)

A study of news comments changes when duplicate posts count multiple times or when certain types of text are excluded. This cell fetches the first 400 rows of the author's public derived table. It keeps text of 8–90 whitespace-separated words, screens for specified contact, markup and sensitive terms, and removes duplicate requests. The saved run retains **222** distinct requests. The filters and the source order define this teaching slice. Compare the sampling rule with the population you would want to describe in a real study.

<details>
<summary>Code from this cell</summary>

```python
# Read and run once. This is an ordered teaching slice of the author's public table.
with ThreadPoolExecutor(max_workers=4) as pool:
    prompt_rows = [
        entry["row"]
        for batch in pool.map(lambda offset: page(WILDCHAT, offset), range(0, 400, 100))
        for entry in batch
    ]

contact_or_markup = re.compile(r"https?://|www\.|[\w.+-]+@[\w.-]+\.\w+|(?:\d[\s()-]*){7,}|<[^>]+>|" + chr(96)*3, re.I)
sensitive = re.compile(r"\b(?:explicit|porn\w*|sexual\w*|rape\w*|incest|suicid\w*|gore|nude|naked|erotic|nsfw|sex|tortur\w*|fetish|orgasm|breasts?|genitals?|molest\w*|kill\w*|murder\w*|blood\w*|drugs?|abuse\w*|nazi\w*)\b", re.I)
def keep_prompt(text):
    return (isinstance(text, str) and 8 <= len(text.split()) <= 90
            and not contact_or_markup.search(text) and not sensitive.search(text))

clean_prompts = [row for row in prompt_rows if keep_prompt(row.get("user_prompt"))]
prompt_table = pd.DataFrame(clean_prompts).drop_duplicates("user_prompt").reset_index(drop=True)
assert len(prompt_table) >= 35, "The returned sample is too small for this activity."
print(f"Fetched {len(prompt_rows)} rows; retained {len(prompt_table)} short, distinct requests.")
```

</details>

### Cell 3 · Find groups and read their terms (`cluster-fit`)

`TfidfVectorizer` builds one feature vector per request. The code excludes common English stop words and keeps terms that appear in at least 3 requests and at most 80% of the sample. K-means assigns requests to nearby centers and updates the centers. In a two-feature illustration, a request at (0.8, 0.2) has squared distance 0.02 from a center at (0.7, 0.3) and 0.72 from one at (0.2, 0.8). It joins the nearer group.

<details>
<summary>Code from this cell</summary>

```python
# Your turn · change K to 3 or 5 after the default run.
K = 4
assert K in {3, 4, 5}

vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.80,
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
X_prompt = vectorizer.fit_transform(prompt_table["user_prompt"])
with threadpool_limits(limits=1):
    km = KMeans(n_clusters=K, n_init=12, random_state=201).fit(X_prompt)
cluster = km.labels_
terms = vectorizer.get_feature_names_out()
top_words = [", ".join(terms[np.argsort(weights)[-5:][::-1]]) for weights in km.cluster_centers_]
clusters = pd.DataFrame({
    "Cluster": np.arange(K),
    "Rows": np.bincount(cluster, minlength=K),
    "Top terms": top_words,
})
display(clusters)

fig, ax = plt.subplots(figsize=(8.0, 3.8), layout="constrained")
bars = ax.barh(clusters["Cluster"].astype(str), clusters["Rows"], color="#137f8c")
ax.bar_label(bars, padding=5)
ax.set_xlim(0, clusters["Rows"].max() * 1.18)
ax.set(xlabel="Requests in the teaching slice", ylabel="Cluster ID", title=f"WildChat-derived request groups  |  K = {K}")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
show_figure(fig)

# Optional check after fitting: overlap with the author's most common source group in each new cluster.
most_common = (
    pd.DataFrame({"New cluster": cluster, "Source cluster": prompt_table["cluster_id"]})
    .value_counts().rename("Overlap rows").reset_index()
    .sort_values(["New cluster", "Overlap rows", "Source cluster"], ascending=[True, False, True])
    .drop_duplicates("New cluster")
)
display(most_common.reset_index(drop=True))
```

</details>

In the saved run, `K = 4`. The table and bar chart show the sizes. Each row of the table also gives five high-weight terms from that group's center.

| Cluster ID | Requests | High-weight terms |
|---:|---:|---|
| 0 | 136 | make, like, character, using, create |
| 1 | 50 | write, story, man, tracer, movie |
| 2 | 22 | minute, scene, dialogue, complex, got |
| 3 | 14 | correct, confidence, answer, rate, scale |

Terms such as *write*, *story* and *movie* suggest a possible theme for cluster 1. The researcher supplies and checks that interpretation. The source's cluster IDs enter an optional overlap table **after** fitting. The fitted groups come from the request text.

### Cell 4 · Read the two views of the result (`explanation-gallery`)

Panel A projects the requests and centers onto two axes. A dot is a request and a large cross marks a center. Group assignments come from the full word vectors. Panel B shows selected term weights for each center, so a darker cell shows a larger weight for that term. The gallery lets an analyst connect a possible group name to the words that support it.

<details>
<summary>Code from this cell</summary>

```python
# Read and run after fitting K-means. This cell makes the two-panel gallery.
def explanation_gallery(filename="STATS201_Week3_Explanation_Gallery.png"):
    with threadpool_limits(limits=1):
        svd = TruncatedSVD(n_components=2, random_state=201).fit(X_prompt)
        projected = svd.transform(X_prompt)
        centers = svd.transform(km.cluster_centers_)
    selected = list(dict.fromkeys(
        int(i) for center in km.cluster_centers_ for i in np.argsort(center)[-2:][::-1]
    ))
    weights = pd.DataFrame(km.cluster_centers_[:, selected],
                           columns=terms[selected], index=np.arange(K))
    palette = sns.color_palette("colorblind", K)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained", facecolor="#f6fbfb")
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=cluster.astype(str),
                    hue_order=[str(j) for j in range(K)], palette=palette,
                    ax=a, alpha=.60, s=26, linewidth=0)
    a.scatter(centers[:, 0], centers[:, 1], c=palette, marker="X",
              s=120, edgecolor="#122838", linewidth=1.0)
    a.set(title=f"A  Projected requests and centers  |  K = {K}",
          xlabel="TF–IDF projection component 1", ylabel="Component 2")
    a.legend(title="Cluster", frameon=False)
    sns.heatmap(weights, cmap="crest", ax=b, linewidths=.5,
                cbar_kws={"label": "Center TF–IDF weight"})
    b.set(title="B  Selected terms at each center", xlabel="Terms", ylabel="Cluster")
    b.tick_params(axis="x", labelrotation=35)
    b.tick_params(axis="y", labelrotation=0)
    fig.suptitle("Unsupervised Explanation  |  WildChat-derived teaching slice",
                 fontsize=16, color="#102b3d", fontweight="bold")
    if filename:
        fig.savefig(filename, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    show_figure(fig)

explanation_gallery()
```

</details>

![Default K=4 gallery with projected WildChat requests and a heatmap of center term weights](figures/week03_02_explanation.png)

**Try one change.** Set `K = 3` or `K = 5` in Cell 3 and rerun Cells 3 and 4. Compare a group's terms and size across runs. Cluster numbers can change when the fit changes. For Ed, share one panel, the setting, your interpretation and a check that could change it.

**Transfer question.** If you grouped public comments about a transport proposal, how might the choice of common words to remove change the themes you report?

**Sources.** The source data are [WildChat, Zhao et al.](https://arxiv.org/abs/2405.01470). The classroom slice comes from [Shen's derived table](https://huggingface.co/datasets/heyyjudes/wildchat-creative-only-labeled). The published analysis describes its representation and grouping in Appendix A.2. [Shen, ICML 2025](https://proceedings.mlr.press/v267/shen25r.html)

