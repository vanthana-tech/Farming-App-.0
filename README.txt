FOLDER: farming app
--------------------
Included (3 files):
  - app.py
  - remedies.py
  - requirements.txt

STILL NEEDED FROM YOU (copy these into this same folder):
  - crop_disease_model.tflite   (exported from your Colab training)
  - labels.json                 (exported from your Colab training)
  - train_disease_model.py      (your training script, if you want it archived here too)
  - sample leaf test images     (for testing each class)

Once all 7 items are in this one folder, run:
  pip install -r requirements.txt
  streamlit run app.py
