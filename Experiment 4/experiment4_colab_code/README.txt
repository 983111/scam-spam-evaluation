# Experiment 4 — Google Colab

## Files
- `Experiment4_Multilingual_Scam_Detection.ipynb`: notebook to upload/open in Google Colab.

## Run
1. Download the notebook and upload it to https://colab.research.google.com/
2. In Colab choose Runtime → Change runtime type → T4 GPU if available.
3. Run each cell from top to bottom.
4. If the schema-inspection cell reports different column names, set TEXT_COL, LABEL_COL, LANG_COL, GROUP_COL or LANG_TEXT_COLS in the mapping cell according to the printed columns.
5. Download the `experiment4_results` folder from the Colab Files pane when finished.

## Critical methodological warning
The primary multilingual SMS dataset was machine-translated from an English SMS corpus. Results on Hindi translations are controlled cross-language tests, not proof of real-world Hindi scam detection. A leakage-safe split requires a shared original-message/group ID. If the dataset does not provide one, do not proceed with claims of leakage-safe evaluation until source-message alignment is available.

No paid APIs are used. Transformer fine-tuning requires internet access to download public datasets and pretrained weights.
