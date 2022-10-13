from google.cloud import storage
from tqdm.std import tqdm
import zipfile
import os
import json

def update_models():
    bucket_name = "ppibot-bucket"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    models = ["TextClassification", "TimeModel", "CountModel", "DataModel"]

    for model in models:
        blob = bucket.get_blob(f"{model}.zip")
        if os.path.isdir(f"./ppinat/models/{model}"):
            data = json.load(open("./ppinat/models/lastUpdateModels.json"))
            last_update = data[model]
            if last_update == blob.updated.strftime("%Y-%m-%d %H:%M:%S.%f+00:00"):
                break
        
        with open(f"./ppinat/models/{model}.zip", 'wb') as f:
            with tqdm.wrapattr(f, "write", total=blob.size, desc=f"Downloading {model}") as file_obj:
                storage_client.download_blob_to_file(blob,file_obj)

        with zipfile.ZipFile(f"./ppinat/models/{model}.zip", 'r') as zip_ref:
            zip_ref.extractall(f"./ppinat/models/")

        os.remove(f"./ppinat/models/{model}.zip")

        data = json.load(open("./ppinat/models/lastUpdateModels.json"))
        data[model] = blob.updated.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

        with open("./ppinat/models/lastUpdateModels.json", "w") as f:
            json.dump(data, f)
