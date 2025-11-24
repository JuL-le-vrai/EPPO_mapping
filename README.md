# EPPO_mapping

Abstract: 
Pests and diseases threaten global crop yields, however the lack of standardized plant damage datasets limits the development of a generalist and robust diagnostic tool. Existing datasets vary widely in naming conventions and scope, limiting interoperability and generalization. To address this, we propose a fully automated method to harmonize plant damage labels across heterogeneous datasets by mapping them to the European and Mediterranean Plant Protection Organization (EPPO) taxonomy. Our approach leverages large language model (LLM) embeddings to capture semantic similarity between label terms, including synonyms, multilingual variants, and vernacular names.

Data handling:
- IPM data used for mapping can be found in the ./data/IPM_data/BD_IPM.csv file. Each row refers to one image in the dataset.
- ePhytia data used for mapping can be found in the ./data/PN_data/PN_import_ePhy_BD.csv file. Each row refers to one image in the dataset. Images can be downloaded through the [Pl@ntNet API](https://my.plantnet.org/).
- CDDM datat used for mapping can be found in the ./data/CDDM_data/BD_CDDM.csv file. Images can be found in the [CDDM github repository](https://github.com/UnicomAI/UnicomBenchmark/tree/main/CDDMBench).
- PlantVillage data used for mapping can be found in the ./data/PV_data/BD_PV.csv file. Images can be found on [kaggle](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage).
- PDILyfspot data used for mapping can be found both in the ./data/Bousset_data/BD_Bousset.csv and ./data/Bousset2_data/BD_Bousset2.csv files. Images are available online under the following DOIs : [10.57745/0U7D1V](https://doi.org/10.57745/0U7D1V), [10.57745/VYR8ZF](https://doi.org/10.57745/VYR8ZF), [10.57745/VFFDDQ](https://doi.org/10.57745/VFFDDQ), [10.57745/D6DVD5](https://doi.org/10.57745/D6DVD5), [10.57745/A6RPFO](https://doi.org/10.57745/A6RPFO), [10.57745/H7JVFN](https://doi.org/10.57745/H7JVFN), [10.57745/OKUEDY](https://doi.org/10.57745/OKUEDY).

Key features:
- Resolve scientific and common names to EPPO codes.
- Batch mapping from CSV inputs.
- Enrich records with taxonomic metadata (taxon rank, family, synonyms).
- Export results to JSON format.



