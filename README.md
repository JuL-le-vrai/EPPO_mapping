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

# Usage

To produce a mapping from a list of names you will need :

- A .json file of the form : `{unique_instance_id (type = str) : [name1 (type = str), name2 (type = str), ...], ...}` containing the diffferent names of the instances you want to map. See the example in `./mapping_module/IPM_names.json`. (If you only have a list of plain text names, build a json from it with lists of one unique element per entry. An integration of this functionnality directly in the code is WIP.)
- The fullcodes.xml file, wich is an export of the EPPO database. The version used for the paper can be found in `./Reproduce_results/data/Extract_EPPO/fullcodes.xml.zip`. Alternatively you can download the latest version at https://data.eppo.int/docs/ (you will need an account)
- An openAI API key for the GPT-based mapping method. Your API account must be provisioned. Notice that the generation of a mapping for 10 000 names is less 1$, most of which coming from the extraction of features for the 50k EPPO codes, that can be reused after generation. 
- A huggingface API key for the other LLM-based methods.

## Basic examples

First You will need to convert the .xml of EPPO codes to a more manageable .json file. For this, run : 
`python xml_to_json.py --EPPO-xml $PATH_TO_EPPO_XML_EXPORT$`
This command will produce a json file named `EPPO_codes.json` which is a json counterpart of the xml export.

To extract a Levenshtein based mapping from your names_file.json run :
`python ./mapping_module/Extract_mapping_from_names.py --tolerance 0.95 --method Levenshtein --names-json $PATH_TO_YOUR_JSON_NAMES_FILE$ --EPPO-json $PATH_TO_EPPO_CODES.JSON$` tolerance parameter can take any value from 0 to 1, 1 being for exact correspondance. Note that if you have different datasets you want to map at once, you can specify a list of json names files paths, comma separated. 

To extract a GPT based mapping from your names_file.json run : 
`python /mapping_module/LLM_FeatureExtraction.py --names-json $PATH_TO_YOUR_JSON_NAMES_FILE$ --GPT-api-key-file $PATH_TO_TXT_FILE_CONTAINNING_API_KEY$ --EPPO-xml $PATH_TO_EPPO_XML_EXPORT$ --model GPT`. This will extract the representative features of the EPPO names and your specified names. Similarly, you can specify a list of json names files paths, comma separated.
Then run :
`python ./mapping_module/Extract_mapping_from_names.py --tolerance $YOUR_SPECIFIED_TOLERANCE$ --method GPT-embed --names-json $PATH_TO_YOUR_JSON_NAMES_FILE$ --EPPO-json $PATH_TO_EPPO_CODES.JSON$`

Resulting mapping can be found in json format in the `./mapping_module/outputs/` directory in a file named `Mapping_$METHOD$_$TOLERANCE$_DStoEPPO_V2.json`

## Output description 

The `Extract_mapping_from_names.py` script will produce two distinct output files : 
- A dictionnary of EPPO codes, wich associates each entry (EPPO code) to : its type in the EPPO terminology, its parent in the EPPO hierarchy, its known names in the EPPO database, and a dictionnary containnnig, for each specified json file, a list of the json entries that map to this code with a similarity above the specified tolerance threshold.
- A Mapping file, in the form of a json file, which, for each specified names file, associates to each entry the best mapping EPPO code, which similarity is above the specified tolerance threshold.