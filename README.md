# Additive Manifesto Decomposition

The code and data in this repository refer to the experiments of the ACL Findings paper: "Additive manifesto decomposition: A policy domain aware method for understanding party positioning" by Tanise Ceron, Dmitry Nikolaev and Sebastian Pado. 

### Paper's abstract: 
"Automatic extraction of party (dis)similarities from texts such as party election manifestos or parliamentary speeches plays an increasing role
in computational political science. However, existing approaches are fundamentally limited to targeting only global party (dis)\-similarity: they condense the relationship between a pair of parties into a single figure, their similarity. In aggregating over all policy domains (e.g., health or foreign policy), they do not provide any qualitative insights into which domains parties agree or disagree on.
This paper proposes a workflow for estimating policy domain aware party similarity that overcomes this limitation. The workflow covers (a)~definition of suitable policy domains; (b)~automatic labeling of domains, if no manual labels are available; (c)~computation of domain-level similarities and aggregation at global level; (d)~extraction of interpretable party positions on major policy axes via multidimensional scaling.
We evaluate our workflow on manifestos from the German federal elections. We find that our method (a)~surpasses existing global models in predicting party similarity and (b)~provides accurate party-specific positions, even with automatically labelled policy domains."


## Code reproduction: 
First create a new environment and install the necessary packages:

    python3 -m venv venv

    source venv/bin/activate

    pip install -r requirements.txt


The first step of the workflow involves grouping the fine-grained MARPOR categories into policy domains. If you want to inspect the different sets of clusters that the method outputs, please run the line below:  

    python3 clustering_policies.py

In order to extract the sentence representations for the analysis later, run:

    python3 get_representations.py

For running the similarity between parties based on text similarity, run the line below. It will run both the similarity with the annotated and predicted data. You can check the results in "./results/de/predicted" and "./results/de/annotated". We make the predictions of all models available in the "./classifier/predictions" folder even though the script in this repository only runs on the "./classifier/predictions" of the best performing model - RoBERTa xml + MLP. The scripts for the classifiers can be found in "./classifiers". The training data can be downloaded from the [Manifesto Project](https://manifesto-project.wzb.eu/).  

    python3 party_similarity.py




Citation below:  

@inproceedings{ceron-additive-2023,
    title = "Additive manifesto decomposition: A policy domain aware method for understanding party positioning",
    author = "Ceron, Tanise  and
      Nikolaev, Dmitry  and
      Pado, Sebastian",
    booktitle = "Findings of the 61st Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics"
}


